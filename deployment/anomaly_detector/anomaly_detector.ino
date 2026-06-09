#include <Arduino.h>
#include <PDM.h>
#include <math.h>

#include "TensorFlowLite.h"
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kAudioSeconds = 1;
constexpr int kAudioSamples = kSampleRate * kAudioSeconds;
constexpr int kPdmChannels = 1;

constexpr int kNfft = 400;
constexpr int kHopLength = 160;
constexpr int kMfccCount = 13;
constexpr int kMelCount = 128;
constexpr int kFftBins = (kNfft / 2) + 1;
constexpr int kFrameCount = 1 + (kAudioSamples / kHopLength);
constexpr int kFeatureChannels = 3;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kAmin = 1.0e-10f;
constexpr float kTopDb = 80.0f;
constexpr float kAnomalyThreshold = 4.727662f;

// CNN structures and their underlying operators use a fair amount of temporary scratch memory. 
// Raised to 128 KB to guarantee no allocation faults during convolution calculations.
constexpr int kTensorArenaSize = 128 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

volatile int samples_read = 0;
int16_t audio_buffer[kAudioSamples];
int16_t pdm_buffer[512];
float power_spectrum[kFftBins];
float mfcc[kFrameCount][kMfccCount];
float delta[kFrameCount][kMfccCount];
float delta2[kFrameCount][kMfccCount];
float cos_step[kFftBins];
float sin_step[kFftBins];
float mel_edges[kMelCount + 2];
float dct_basis[kMfccCount][kMelCount];
float window_fn[kNfft];

float hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void onPDMdata() {
  int bytes_available = PDM.available();
  if (bytes_available <= 0) {
    return;
  }

  int bytes_to_read = min(bytes_available, static_cast<int>(sizeof(pdm_buffer)));
  PDM.read(pdm_buffer, bytes_to_read);

  int new_samples = bytes_to_read / static_cast<int>(sizeof(int16_t));
  int room = kAudioSamples - samples_read;
  int samples_to_copy = min(new_samples, room);

  for (int i = 0; i < samples_to_copy; ++i) {
    audio_buffer[samples_read + i] = pdm_buffer[i];
  }
  samples_read += samples_to_copy;
}

void prepareFeatureTables() {
  for (int k = 0; k < kFftBins; ++k) {
    float angle = -2.0f * kPi * static_cast<float>(k) / static_cast<float>(kNfft);
    cos_step[k] = cosf(angle);
    sin_step[k] = sinf(angle);
  }

  for (int n = 0; n < kNfft; ++n) {
    window_fn[n] = 0.5f - 0.5f * cosf((2.0f * kPi * n) / static_cast<float>(kNfft - 1));
  }

  float mel_min = hzToMel(0.0f);
  float mel_max = hzToMel(kSampleRate / 2.0f);
  for (int i = 0; i < kMelCount + 2; ++i) {
    float mel = mel_min + (mel_max - mel_min) * i / static_cast<float>(kMelCount + 1);
    mel_edges[i] = melToHz(mel);
  }

  float dct_scale0 = sqrtf(1.0f / static_cast<float>(kMelCount));
  float dct_scale = sqrtf(2.0f / static_cast<float>(kMelCount));
  for (int c = 0; c < kMfccCount; ++c) {
    float scale = (c == 0) ? dct_scale0 : dct_scale;
    for (int m = 0; m < kMelCount; ++m) {
      dct_basis[c][m] = scale * cosf(kPi * c * (static_cast<float>(m) + 0.5f) /
                                    static_cast<float>(kMelCount));
    }
  }
}

void computePowerSpectrumForFrame(int frame_index) {
  int frame_start = frame_index * kHopLength - (kNfft / 2);

  for (int k = 0; k < kFftBins; ++k) {
    float wr = 1.0f;
    float wi = 0.0f;
    float real = 0.0f;
    float imag = 0.0f;
    float c = cos_step[k];
    float s = sin_step[k];

    for (int n = 0; n < kNfft; ++n) {
      int sample_index = frame_start + n;
      float sample = 0.0f;
      if (sample_index >= 0 && sample_index < kAudioSamples) {
        sample = static_cast<float>(audio_buffer[sample_index]) / 32768.0f;
      }

      sample *= window_fn[n];
      real += sample * wr;
      imag += sample * wi;

      float next_wr = wr * c - wi * s;
      wi = wr * s + wi * c;
      wr = next_wr;
    }

    power_spectrum[k] = real * real + imag * imag;
  }
}

void melFilterEnergy(int mel_index, float* energies) {
  float left = mel_edges[mel_index];
  float center = mel_edges[mel_index + 1];
  float right = mel_edges[mel_index + 2];
  float energy = 0.0f;

  for (int bin = 0; bin < kFftBins; ++bin) {
    float hz = (static_cast<float>(bin) * kSampleRate) / static_cast<float>(kNfft);
    float weight = 0.0f;

    if (hz >= left && hz <= center) {
      weight = (hz - left) / (center - left);
    } else if (hz > center && hz <= right) {
      weight = (right - hz) / (right - center);
    }

    energy += power_spectrum[bin] * weight;
  }
  *energies = max(energy, kAmin);
}

void computeMfcc() {
  float log_mel[kMelCount];

  for (int frame = 0; frame < kFrameCount; ++frame) {
    computePowerSpectrumForFrame(frame);

    float max_db = -100000.0f;
    for (int m = 0; m < kMelCount; ++m) {
      melFilterEnergy(m, &log_mel[m]);
      log_mel[m] = 10.0f * log10f(log_mel[m]);
      if (log_mel[m] > max_db) {
        max_db = log_mel[m];
      }
    }

    float floor_db = max_db - kTopDb;
    for (int m = 0; m < kMelCount; ++m) {
      log_mel[m] = max(log_mel[m], floor_db);
    }

    for (int c = 0; c < kMfccCount; ++c) {
      float coeff = 0.0f;
      for (int m = 0; m < kMelCount; ++m) {
        coeff += dct_basis[c][m] * log_mel[m];
      }
      mfcc[frame][c] = coeff;
    }
  }
}

void computeDelta(const float src[kFrameCount][kMfccCount], float dst[kFrameCount][kMfccCount]) {
  constexpr int width = 9;
  constexpr int half_width = width / 2;
  constexpr float denom = 60.0f;

  for (int t = 0; t < kFrameCount; ++t) {
    for (int c = 0; c < kMfccCount; ++c) {
      float value = 0.0f;
      for (int n = 1; n <= half_width; ++n) {
        int plus = min(kFrameCount - 1, t + n);
        int minus = max(0, t - n);
        value += n * (src[plus][c] - src[minus][c]);
      }
      dst[t][c] = value / denom;
    }
  }
}

int8_t quantizeInput(float value) {
  int32_t quantized = static_cast<int32_t>(roundf(value / input->params.scale)) + input->params.zero_point;
  quantized = min(127, max(-128, quantized));
  return static_cast<int8_t>(quantized);
}

float dequantizeValue(int8_t quantized, float scale, int32_t zero_point) {
  return (static_cast<int32_t>(quantized) - zero_point) * scale;
}

bool fillModelInput() {
  // Validate that the model shape matches our expected height, width, and channels dimensions
  if (input->dims->size != 4) {
    Serial.println("Error: Expected a 4D input tensor [Batch, Height, Width, Channels]");
    return false;
  }
  if (input->dims->data[1] != kFrameCount || input->dims->data[2] != kMfccCount || input->dims->data[3] != kFeatureChannels) {
    Serial.println("Error: Model dimension tracking mismatch.");
    return false;
  }

  computeMfcc();
  computeDelta(mfcc, delta);
  computeDelta(delta, delta2);

  // 2D CNN tensor mapping
  // Access memory as index = ((f * kMfccCount + c) * kFeatureChannels) + channel
  for (int frame = 0; frame < kFrameCount; ++frame) {
    for (int coeff = 0; coeff < kMfccCount; ++coeff) {
      int base_offset = ((frame * kMfccCount) + coeff) * kFeatureChannels;
      
      input->data.int8[base_offset + 0] = quantizeInput(mfcc[frame][coeff]);
      input->data.int8[base_offset + 1] = quantizeInput(delta[frame][coeff]);
      input->data.int8[base_offset + 2] = quantizeInput(delta2[frame][coeff]);
    }
  }

  return true;
}

void clearAudioBuffer() {
  noInterrupts();
  samples_read = 0;
  interrupts();
}

bool hasFullAudioBuffer() {
  noInterrupts();
  bool full = samples_read >= kAudioSamples;
  interrupts();
  return full;
}

}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }

  prepareFeatureTables();

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema version mismatch.");
    while (true) {
      delay(1000);
    }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed. Try increasing kTensorArenaSize.");
    while (true) {
      delay(1000);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type != kTfLiteInt8) {
    Serial.println("This sketch expects an int8 input model.");
    while (true) {
      delay(1000);
    }
  }

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(kPdmChannels, kSampleRate)) {
    Serial.println("Failed to start PDM microphone.");
    while (true) {
      delay(1000);
    }
  }

  PDM.setGain(30);
  clearAudioBuffer();
  Serial.println("ready");
}

void loop() {
  if (!hasFullAudioBuffer()) {
    delay(1);
    return;
  }

  PDM.end();

  // 1. Populate the input tensor with your MFCC / Delta features
  if (fillModelInput()) {
    
    int total_elements = kFrameCount * kMfccCount * kFeatureChannels;
    
    // 2. CRITICAL: Take a physical snapshot of the input BEFORE invoking the model
    // This protects your data from being overwritten by TFLite's buffer reuse optimization.
    static int8_t baseline_input_snapshot[kFrameCount * kMfccCount * kFeatureChannels];
    memcpy(baseline_input_snapshot, input->data.int8, total_elements * sizeof(int8_t));

    // 3. Run the model inference safely
    if (interpreter->Invoke() == kTfLiteOk) {
      float total_squared_error = 0.0f;
      
      float input_scale = input->params.scale;
      int32_t input_zero_point = input->params.zero_point;
      
      float output_scale = output->params.scale;
      int32_t output_zero_point = output->params.zero_point;

      for (int i = 0; i < total_elements; ++i) {
        // 1. Dequantize the saved input snapshot using the model's actual scale
        // This scales your large Arduino features down to match Python's training range
        float original = (static_cast<float>(baseline_input_snapshot[i]) - input_zero_point) * input_scale;
        
        // 2. Safely extract and dequantize the output tensor byte
        float reconstructed = 0.0f;
        if (output->type == kTfLiteFloat32) {
          reconstructed = output->data.f[i];
        } else {
          reconstructed = (static_cast<float>(output->data.int8[i]) - output_zero_point) * output_scale;
        }
        
        // 3. Compute standard squared error deviation
        float error = original - reconstructed;
        total_squared_error += error * error;
      }

      float anomaly_score = total_squared_error / static_cast<float>(total_elements);

      Serial.print("Anomaly Score (MSE): ");
      Serial.println(anomaly_score, 4);
      Serial.println(anomaly_score >= kAnomalyThreshold ? "ANOMALY DETECTED (yes)" : "NORMAL (no)");
    } else {
      Serial.println("Error: Interpreter Invoke failed.");
    }
  }

  clearAudioBuffer();
  PDM.begin(kPdmChannels, kSampleRate);
}