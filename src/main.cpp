//#include <Arduino.h>


#include <Wire.h>              //for ESP8266 use bug free i2c driver https://github.com/enjoyneering/ESP8266-I2C-Driver
#include <LiquidCrystal_I2C.h>

#include <Adafruit_NeoPixel.h>

LiquidCrystal_I2C lcd(0x27, 16, 4);

#include "Arduino.h"
//#define DEFAULT 1  in above  WTF


// How many NeoPixels are attached to the Arduino?
#define LED_COUNT 3
#define LED_PIN   2
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

#include "model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// See http://www.vlsi.fi/fileadmin/datasheets/vs1053.pdf Pg 31
#define VS1053_BANK_DEFAULT 0x00
#define VS1053_BANK_DRUMS1 0x78
#define VS1053_BANK_DRUMS2 0x7F
#define VS1053_BANK_MELODY 0x79

// See http://www.vlsi.fi/fileadmin/datasheets/vs1053.pdf Pg 32 for more!
// 1 to 128
#define VS1053_GM1_OCARINA 80

#define MIDI_NOTE_ON  0x90
#define MIDI_NOTE_OFF 0x80
#define MIDI_CHAN_MSG 0xB0
#define MIDI_CHAN_BANK 0x00
#define MIDI_CHAN_VOLUME 0x07
#define MIDI_CHAN_PROGRAM 0xC0

#define VS1053_MIDI Serial2
// use GPIO17 U2_TXD on esp32
#define VS1053_RESET 4 // This is the pin that connects to the RESET pin on VS1053


//---------------------
// model.cpp is defined as below. consistent with model.h
// #include "model.h"
// Keep model aligned to 8 bytes to guarantee aligned 64-bit accesses.
// alignas(8) const unsigned char micro_tflite[] = {
// const int micro_tflite_len = 92548;

//  Hybrid models are not supported on TFLite Micro.
// cannot use default quantization


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// size is trial and error
// model input, output and intermediate tensors
//constexpr int kTensorArenaSize = 1024*101;

const int kTensorArenaSize = 100000;
// RAM:   [====      ]  37.8% (used 123944 bytes from 327680 bytes)
// 105000 region `dram0_0_seg' overflowed by 368 bytes
uint8_t tensor_arena[kTensorArenaSize];

}  // namespace


// --------------------------------------------------------
// include corpus (int index) and dictionary int to string
// --------------------------------------------------------
#include "corpus.h"  // C source file automatically generated. DEFINITION (not only declaration)
#include "dictionary.h" // C source file automatically generated. DEFINITION (not only declaration)

const int seqlen = 10*10; // could get dynamically from model
int input_int[seqlen]; // input array, array of int index into dictionary
int dummy[seqlen];

//char buffer[50];

int seed; // 1st index in seed 

int midi_code[5]; // set by parse function with midi code for chords
int midi_len = 0 ; // set by parse function with number of midi code
const int midi_base = 60; // default octave for chords in normal mode
int duration = 200; // ms

int adc;
#define adc_pin 36
int inst;

const boolean real_time = false;

const int to_play_nb = 50; // nb of notes or chords to predict before rendering
int to_play_index = 0; // index in array

// all midi code to play. use special char to separate
int to_play[5*to_play_nb]; // account for separator, chords

PROGMEM const char *instrument[] =  {"piano", "occarina", "plause" };

void midiSetInstrument(uint8_t chan, uint8_t inst) {
  if (chan > 15) return;
  inst --; // page 32 has instruments starting with 1 not 0 :(
  if (inst > 127) return;
  
  VS1053_MIDI.write(MIDI_CHAN_PROGRAM | chan);  
  VS1053_MIDI.write(inst);
}


void midiSetChannelVolume(uint8_t chan, uint8_t vol) {
  if (chan > 15) return;
  if (vol > 127) return;
  
  VS1053_MIDI.write(MIDI_CHAN_MSG | chan);
  VS1053_MIDI.write(MIDI_CHAN_VOLUME);
  VS1053_MIDI.write(vol);
}

void midiSetChannelBank(uint8_t chan, uint8_t bank) {
  if (chan > 15) return;
  if (bank > 127) return;
  
  VS1053_MIDI.write(MIDI_CHAN_MSG | chan);
  VS1053_MIDI.write((uint8_t)MIDI_CHAN_BANK);
  VS1053_MIDI.write(bank);
}

void midiNoteOn(uint8_t chan, uint8_t n, uint8_t vel) {
  if (chan > 15) return;
  if (n > 127) return;
  if (vel > 127) return;
  
  VS1053_MIDI.write(MIDI_NOTE_ON | chan);
  VS1053_MIDI.write(n);
  VS1053_MIDI.write(vel);
}

void midiNoteOff(uint8_t chan, uint8_t n, uint8_t vel) {
  if (chan > 15) return;
  if (n > 127) return;
  if (vel > 127) return;
  
  VS1053_MIDI.write(MIDI_NOTE_OFF | chan);
  VS1053_MIDI.write(n);
  VS1053_MIDI.write(vel);
}


// ------------------------------------------------------------
// parse pitch char* into list of int MIDI code. set global array
// ------------------------------------------------------------
void midi_parse(char *input) {
  
//TF_LITE_REPORT_ERROR(error_reporter, "char* to parse %s" , input);

String pi = input; // convert char* to String object
int len = pi.length();
int octave;

// look for . signature of chords
int first_dot = pi.indexOf('.');
int last_dot = pi.lastIndexOf('.');

if (first_dot == -1) { // this is not a chord  could be "0" or "B-4", "B2",
  //TF_LITE_REPORT_ERROR(error_reporter, "parsing single note");
  // C1 = 24, A1 = 36 , 12 pitches in between

    int code;
    if (input[0] == 'C') code =24;
    if (input[0] == 'D') code =26;
    if (input[0] == 'E') code =28;
    if (input[0] == 'F') code =29;
    if (input[0] == 'G') code =31; // G1
    if (input[0] == 'A') code =33;
    if (input[0] == 'B') code =35;
  
  if (len == 2) { //  convert B2 into midi code
    octave = input[1] - '1' ; //  starts at octave 1
    code = code + octave * 12;  
  }

  else if (len == 3) { // convert C#2 into midi code
    octave = input[2] - '1';
    if (input[1] == '#') code = code +1; 
    if (input[1] == '-') code = code -1; 
    code = code + octave * 12;  
  }

  else if (len == 1){ // REST or "0"
    if (input[0] == 'R') { 
      //TF_LITE_REPORT_ERROR(error_reporter, "parsing REST");
      code = -1;
      octave = 4;  
    } // R
    else {
     //TF_LITE_REPORT_ERROR(error_reporter, "parsing '0' type ");
    code = input[0] - '0' + midi_base;
    octave = 4;
  } // "0"

  } // len = 1

  else {
    TF_LITE_REPORT_ERROR(error_reporter, "ERROR PARSING");
    return;
  }

  midi_code[0] = code;
  midi_len = 1;
  //TF_LITE_REPORT_ERROR(error_reporter, "parsed single note: octave %d code %d" , octave, code);
  
} // single note

else { // this is a chord
  //TF_LITE_REPORT_ERROR(error_reporter, "parsing chord");
  
  if (first_dot == last_dot) { // chord with only 2 notes
    //TF_LITE_REPORT_ERROR(error_reporter, "2 notes chord. first_dot %d, last_dot %d" , first_dot, last_dot);
    String note1 = pi.substring(0,first_dot);
    String note2 = pi.substring(first_dot+1);

    // NORMAL mode convert String (ie one char) into midi code
    int code0 = note1.toInt() + midi_base;  // NORMAL mode
    int code1 = note2.toInt() + midi_base;
    //TF_LITE_REPORT_ERROR(error_reporter, "midi code %d %d" , code0, code1);

    // fill global array
    midi_len = 2;
    midi_code[0] = code0;
    midi_code[1] = code1;
    
  } // 2 notes
 
  else { // chord with more than 2 notes

    //TF_LITE_REPORT_ERROR(error_reporter, "3 or more notes chord. first_dot %d, last_dot %d" , first_dot, last_dot);

    boolean done = false;
    byte number_of_notes = 0;  // also index in array
    
    int dot;
    int start;
    int eend;
    String note;

    // first note in chord
    start = 0;
    eend = first_dot;
    
    while (done == false) {

      note = pi.substring(start,eend);
     // fill array of midi code
      midi_code[number_of_notes] = note.toInt() + midi_base;
      //TF_LITE_REPORT_ERROR(error_reporter, "filling array index %d, code %d" , number_of_notes, midi_code[number_of_notes]);
      number_of_notes ++;

      // see if there is another note in chord
      dot = pi.indexOf('.', eend+1); // value, from
      if (dot == -1) { done = true;
      // process last note
      note = pi.substring(eend+1);
      Serial.print("last note: ");
      Serial.println(note);
      midi_code[number_of_notes] = note.toInt() + midi_base;
      //TF_LITE_REPORT_ERROR(error_reporter, "filling array index %d, code %d" , number_of_notes, midi_code[number_of_notes]);
      number_of_notes ++;
      
      } else {
        start = eend+1;
        eend = dot;
        done = false; 
      }
    } // while
    
    midi_len = number_of_notes;
 
  } // more than 2 notes
  
} // chord

  // display final parsing result
  for (int i =0; i<midi_len; i++) {
  //TF_LITE_REPORT_ERROR(error_reporter, "PARSING RESULT: GLOBAL array len %d %d" , midi_len, midi_code[i]);
    
  }
} // parse


void setup() {

  Serial.begin(9600);
  delay(5000);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  TF_LITE_REPORT_ERROR(error_reporter, "\n\nPABOU BACH starting");

  // ADC 0 to 3490
  adc = analogRead(adc_pin);
  TF_LITE_REPORT_ERROR(error_reporter, "ADC %d", adc);

  TF_LITE_REPORT_ERROR(error_reporter, "create LCD");
  lcd.begin();
  lcd.backlight();
  lcd.clear();

  //The cursor is at 4th column(count from 0), and 0th row(count from 0).
  lcd.setCursor(0,0);
  lcd.print("TF lite micro");
  lcd.setCursor(0,1);
  lcd.print("micro bach");
 
  lcd.setCursor(0,3);
  lcd.print("instrument");
  //lcd.setCursor(12,3);
  //lcd.print(adc);

  TF_LITE_REPORT_ERROR(error_reporter, "create neopixels");
  strip.begin();           // INITIALIZE NeoPixel strip object (REQUIRED)
  strip.show();            // Turn OFF all pixels ASAP
  strip.setBrightness(50); // Set BRIGHTNESS to about 1/5 (max = 255)

  // test neopixels

  
  for(int i=0; i<strip.numPixels(); i++) { // For each pixel in strip...
    strip.setPixelColor(i, strip.Color(0,   255,   0));         //  Set pixel's color (in RAM)
    strip.show();                          //  Update strip to match
    delay(1000);                           //  Pause for a moment
    strip.clear();
  }
   strip.clear();
   strip.show();   


// ADC 0 to 3490
int x;

x = analogRead(36);
TF_LITE_REPORT_ERROR(error_reporter, "ADC %d", x);
  
  VS1053_MIDI.begin(31250); // MIDI uses a 'strange baud rate'

  // MIDI init
  pinMode(VS1053_RESET, OUTPUT);
  digitalWrite(VS1053_RESET, LOW);
  delay(10);
  digitalWrite(VS1053_RESET, HIGH);
  delay(10);

  midiSetChannelBank(0, VS1053_BANK_MELODY);
  midiSetInstrument(0, VS1053_GM1_OCARINA);
  midiSetChannelVolume(0, 127);

  TF_LITE_REPORT_ERROR(error_reporter, "VS1053 started");

  // dictionary

  TF_LITE_REPORT_ERROR(error_reporter, "dictionary:");
   for (int i =0; i < dictionary_len; i++) {
    //TF_LITE_REPORT_ERROR(error_reporter, dictionary[i]);
   }


  //for (int i =0; i < corpus_len; i++) {
  //  Serial.println(corpus[i]);
  //}

// ---------------------------------------------------
// for testing, parsing all dictionary
// ---------------------------------------------------
  Serial.println("parsing dictionary");
  //for (int i=0; i< dictionary_len; i++) {
   //TF_LITE_REPORT_ERROR(error_reporter, "\ntest: will parse #%d, %s", i, dictionary[i]);
   //midi_parse(dictionary[i]); 
  //}
  Serial.println("DONE parsing dictionary");
  

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  // hex is in model.cpp
  // parameter is name of array alignas(8) const unsigned char g_model[] = {
  model = tflite::GetModel(micro_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "\nmodel loaded");

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // define micromutableops to only select needed ops. see audio example. to save memory
  TF_LITE_REPORT_ERROR(error_reporter, "use ALL ops");
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
   TF_LITE_REPORT_ERROR(error_reporter, "create interpreter");
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  TF_LITE_REPORT_ERROR(error_reporter, "allocate tensors");
  // Allocate memory from the tensor_arena for the model's tensors.
  // 12 bytes lost due to alignment. To avoid this loss, please make sure the tensor_arena is 16 bytes aligned.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  // fp32
  // Arena size is too small for all buffers. Needed 102400 but only 98080 was available.
  
  // fp16 GPU Flash: [=====     ]  52.5% (used 2067558 bytes from 3936256 bytes)
  // Node DEQUANTIZE (number 1f) failed to prepare with status 1

  // 16x8 Flash: [===       ]  34.9% (used 1371878 bytes from 3936256 bytes)
  //lib\tfmicro\tensorflow\lite\micro\kernels\fully_connected.cc Hybrid models are not supported on TFLite Micro.
  //Node FULLY_CONNECTED (number 12f) failed to prepare with status 1

  //TPU Flash: [===       ]  34.7% (used 1366086 bytes from 3936256 bytes)
  /*
input dim size 3 
input type 9 
input type int8
input dim 1 size 1 
input dim 2 size 10 
input dim 3 size 10 
output dim size 2 
output type 9 
output type int8
output dim 1 size 1 
output dim 2 size 95 
  */

  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "tensors allocated");

  //lib\tfmicro\tensorflow\lite\micro\kernels\quantize.cc:69 input->type == kTfLiteFloat32 || input->type == kTfLiteInt16 || input->type == kTfLiteInt8 was not true.

  // Obtain pointers to the model's input and output tensors.
  // only one input there
  input = interpreter->input(0);
  output = interpreter->output(0);

  // input  data, dim, type, params (quantization)

// dim is tensor shape , one element for each dim, value of each is len of tensor

TF_LITE_REPORT_ERROR(error_reporter, "input dim size %d ", input->dims->size); // size is 2, dummy dim , len of shape
TF_LITE_REPORT_ERROR(error_reporter, "input type %d ", input->type); // 9
if (input->type == kTfLiteFloat32) {TF_LITE_REPORT_ERROR(error_reporter, "input type float32", input->type);}
if (input->type == kTfLiteInt8) {TF_LITE_REPORT_ERROR(error_reporter, "input type int8", input->type);} // 
TF_LITE_REPORT_ERROR(error_reporter, "input dim 1 size %d ", input->dims->data[0]); //  shape[0]
TF_LITE_REPORT_ERROR(error_reporter, "input dim 2 size %d ", input->dims->data[1]); // 1 shape[1] , ie scalar
TF_LITE_REPORT_ERROR(error_reporter, "input dim 3 size %d ", input->dims->data[2]); // 1 shape[1] , ie scalar

TF_LITE_REPORT_ERROR(error_reporter, "output dim size %d ", output->dims->size);
TF_LITE_REPORT_ERROR(error_reporter, "output type %d ", input->type);
if (output->type == kTfLiteFloat32) {TF_LITE_REPORT_ERROR(error_reporter, "input type float32", output->type);}
if (output->type == kTfLiteInt8) {TF_LITE_REPORT_ERROR(error_reporter, "output type int8", output->type);}
TF_LITE_REPORT_ERROR(error_reporter, "output dim 1 size %d ", output->dims->data[0]);
TF_LITE_REPORT_ERROR(error_reporter, "output dim 2 size %d ", output->dims->data[1]);

TF_LITE_REPORT_ERROR(error_reporter, "end of setup");

} // setup

void loop() {
  boolean new_seed = false;
  char *pi;

  // ------------------------------------------------------------------
  // real time
  // ------------------------------------------------------------------

  if(real_time) {
    // get random seed. A long is an integer. No conversion necessary
    seed = random(corpus_len-seqlen-1);
    seed = 0; // to test
    TF_LITE_REPORT_ERROR(error_reporter, "RT. random seed index %d ", seed);

    // fill global input_int array with corpus, starting at seed index
    for (int i=0; i< seqlen; i++) {
      input_int[i] = corpus[seed];
      seed = seed + 1;
    }

    //-------------------------------------------------------------
    // play seed 
    //--------------------------------------------------------------
    // convert int to string, parse string into to individual pitches and duration, convert to midi code and play midi code

    for (int i=seed; i< seqlen; i++) { // play seed
    // convert int into String (or char array);
    pi = dictionary[corpus[i]]; 
    TF_LITE_REPORT_ERROR(error_reporter, "RT. play seed note %s ", pi);

      // parse String, fill midi_code array 
      midi_parse(pi); 
    
      // play each individual midi code
      for (int j=0; j<midi_len; j++) {
        if (midi_code[j] != -1) // not a rest
          midiNoteOn(0, midi_code[j], 127);
      }
      
      delay(duration); // duration
      
      for (int j=0; j<midi_len; j++) {
        if (midi_code[j] != -1)
          midiNoteOff(0, midi_code[j], 127);
      }
    } // play seed 

    // input_int is an int array, and is up to date

    // ----------------------------------------------------------------
    // predict 'forever'
    // ----------------------------------------------------------------
    while (true) {

   if (input->type == kTfLiteInt8)  {
     TF_LITE_REPORT_ERROR(error_reporter, "input is int 8");

        // Quantize the input from floating-point to integer
        //int8_t x_quantized = x / input->params.scale + input->params.zero_point;
        // Place the quantized input in the model's input tensor
        //input->data.int8[0] = x_quantized;


    // fill TFlite input area with QUANTIZED input_int array
    for (int i=0; i< seqlen; i++) {
      int8_t x_quantized = input_int[i] / input->params.scale + input->params.zero_point;
      input->data.int8[i] = x_quantized;
    }

   } // int8

   else {

    //TF_LITE_REPORT_ERROR(error_reporter, "input is fp32");

    // model was trained with X normalized
    float X_mean = 72.97;
    float X_std = 14.58;
    float x;

    // fill TFlite input area with input_int array
    for (int i =0; i< seqlen; i++) {
      x = float(input_int[i]);
      x = x - X_mean;
      x = x / X_std;
      input->data.f[i] = x; // convert int to FLOAT
  }

  } // fp32


  // Run inference, and report any error
  //TF_LITE_REPORT_ERROR(error_reporter, "!!!!!! run inference");
  //unsigned long inference;
  //inference = millis();

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  //TF_LITE_REPORT_ERROR(error_reporter, "Inference ms %f", millis() - my_time);
  // TF use power of 2

  //sprintf(buffer, "inference %d ms", (millis()-inference));
  //Serial.println(buffer);
  // 1550ms with fp32

  int softmax = output->dims->data[1];
  //TF_LITE_REPORT_ERROR(error_reporter, "output softmax size %d", softmax);

  // get argmax, ie index of maximum in softmax
  int argmax = 0;
  float v =0.0;

  for (int i =0; i< softmax; i++) {
    //TF_LITE_REPORT_ERROR(error_reporter, "rank %d, proba %f, argmax %d , max proba %f ",i, output->data.f[i], argmax, v);
    if (output->data.f[i] > v)  { argmax = i; v = output->data.f[i]; }
  }
 
  
  // Obtain the quantized output from model's output tensor
  //int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  //float y = (y_quantized - output->params.zero_point) * output->params.scale;
  
  // no need to dequantize output. it's a fp32 softmax

  // get predicted char * from dictionary
  char *  pred = dictionary[argmax]; // char *
  //TF_LITE_REPORT_ERROR(error_reporter, "pred %s argmax %d ", pred, argmax);

  // parse char *, fill midi_code array and midi_len
  midi_parse(pred); 
  
  // play each individual midi code
  for (int j=0; j<midi_len; j++) {
    if (midi_code[j] != -1)
   midiNoteOn(0, midi_code[j], 127);
  }

  delay(duration); // duration

  for (int j=0; j<midi_len; j++) {
    if (midi_code[j] != -1)
   midiNoteOff(0, midi_code[j], 127);
  }
 
  // update input array
  for (int x=0; x<seqlen-1; x++) {
    dummy[x] = input_int[x+1];
  }
  dummy[seqlen-1] = argmax; // update input with prediction
  for (int y=0; y<seqlen; y++) {input_int[y] = dummy[y];}
  
   //TF_LITE_REPORT_ERROR(error_reporter, "ROLLOVER: %d %d %d; %d %d %d", input_int[0], input_int[1], input_int[2], input_int[seqlen-3],input_int[seqlen-2], input_int[seqlen-1]);
  
    } // while true
  } // if real time


  // ---------------------------------------------------------------
  // NON real time
  // ---------------------------------------------------------------
  if(!real_time) {

  /*
  loop()
   while
    create seed
    run n inference. takes time
      store result in array with separator
    play seed
    render prediction
    exit while to restart from new seed 
  */


  while (!new_seed) {

  // get random seed. A long is an integer. No conversion necessary
  seed = random(corpus_len-seqlen-1);
  seed = 0; // to test

  int x = seed; // keep seed value intact. needed later
  TF_LITE_REPORT_ERROR(error_reporter, "BATCH. random seed index %d ", seed);

  // empty midi code array
  to_play_index =0;

  // fill global input_int array with corpus, starting at seed index
  for (int i=0; i< seqlen; i++) {
    input_int[i] = corpus[x];
    x = x + 1;
  }

  // input_int is an int array, and is up to date
  // run n inferences

    for (int n=0; n< to_play_nb ; n++) {

    TF_LITE_REPORT_ERROR(error_reporter, "BATCH inference: %d %d", n, to_play_index);

   if (input->type == kTfLiteInt8)  {
     //TF_LITE_REPORT_ERROR(error_reporter, "input is int 8");

    // fill TFlite input area with QUANTIZED input_int array
    for (int i=0; i< seqlen; i++) {
      int8_t x_quantized = input_int[i] / input->params.scale + input->params.zero_point;
      input->data.int8[i] = x_quantized;
    }

   } // int8

   else {

    //TF_LITE_REPORT_ERROR(error_reporter, "input is fp32");
    // model was trained with X normalized
    float X_mean = 72.97;
    float X_std = 14.58;
    float x;

    // fill TFlite input area with input_int array
    for (int i =0; i< seqlen; i++) {
      x = float(input_int[i]);
      x = x - X_mean;
      x = x / X_std;
      input->data.f[i] = x; // convert int to FLOAT
  }

  } // fp32

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  
  int softmax = output->dims->data[1];

  // get argmax, ie index of maximum in softmax
  int argmax = 0;
  float v =0.0;

  for (int i =0; i< softmax; i++) {
    //TF_LITE_REPORT_ERROR(error_reporter, "rank %d, proba %f, argmax %d , max proba %f ",i, output->data.f[i], argmax, v);
    if (output->data.f[i] > v)  { argmax = i; v = output->data.f[i]; }
  }
 
  // get predicted char * from dictionary
  char *  pred = dictionary[argmax]; // char *
  //TF_LITE_REPORT_ERROR(error_reporter, "pred %s argmax %d ", pred, argmax);

  // parse char *, fill midi_code array and midi_len
  midi_parse(pred);   

  // store predicted midi code(s) in an array. use -2 as separator

    for (int j=0; j<midi_len; j++) { // store one note or chord
      to_play[to_play_index] = midi_code[j];
      to_play_index = to_play_index +1;
    }
    to_play[to_play_index] = -2; // add separator
    to_play_index = to_play_index +1;

    // update input array for next prediction
      for (int x=0; x<seqlen-1; x++) {
        dummy[x] = input_int[x+1];
      }
      dummy[seqlen-1] = argmax; // update input with prediction
      for (int y=0; y<seqlen; y++) {input_int[y] = dummy[y];}
      
      //TF_LITE_REPORT_ERROR(error_reporter, "ROLLOVER: %d %d %d; %d %d %d", input_int[0], input_int[1], input_int[2], input_int[seqlen-3],input_int[seqlen-2], input_int[seqlen-1]);

    } // run n inferences

    TF_LITE_REPORT_ERROR(error_reporter, "batch: all inference done. time to render %d" , to_play_index);

    // map adc to instrument
      adc = analogRead(adc_pin);
      inst = map(analogRead(adc_pin), 0, 4096, 1, 128 );

      TF_LITE_REPORT_ERROR(error_reporter, "BATCH. mapping %d to %d ", adc, inst);

      // set instrument 
      midiSetInstrument(0, inst);
      midiSetChannelVolume(0, 100);
      lcd.setCursor(12,3);
      lcd.print(inst);
    
    //-------------------------------------------------------------
    // play seed 
    //--------------------------------------------------------------
    // convert int to string, parse string into to individual pitches and duration, convert to midi code and play midi code

    TF_LITE_REPORT_ERROR(error_reporter, "batch: play seed index %d seqlen %d" , seed, seqlen);
    for (int i=seed; i< seqlen; i++) { // play seed
      // convert int into String (or char array);
      pi = dictionary[corpus[i]]; 
      TF_LITE_REPORT_ERROR(error_reporter, "BATCH. play seed note %s ", pi);

      // parse String, fill midi_code array 
      midi_parse(pi); 
    
      // play each individual midi code
      for (int j=0; j<midi_len; j++) {
        if (midi_code[j] != -1) // not a rest
          midiNoteOn(0, midi_code[j], 127);
      }
      
      delay(200); // duration
      
      for (int j=0; j<midi_len; j++) {
        if (midi_code[j] != -1)
          midiNoteOff(0, midi_code[j], 127);
      }
    } // play seed 


      // ---------------------------------------------------------------
      // rendering prediction
      // ---------------------------------------------------------------

      // to_play is an array of midi code, separated by -2;

      int len = to_play_index; // of array created
      to_play_index = 0; // read array from begining
      int code;
      int off[5]; // remember the note that are on
      int off_index = 0;

      // render all midi codes in to_play array
      int color;
      int pixel;

      while (to_play_index < len) { // scan array of midi code
        code = to_play[to_play_index];
        to_play_index = to_play_index +1;

        TF_LITE_REPORT_ERROR(error_reporter, "code %d index %d", code, to_play_index);

        // test array content for midi code, separator or rest

        if (code != -1 and code != -2) { // not a rest or a separator

        // use index to drive neopixels
        if ((to_play_index % 3) == 0) {
        color = strip.Color(255, 0, 0);
        pixel =0;
        }

        if ((to_play_index % 3) == 1) {
        color = strip.Color(0, 255, 0);
        pixel =1;
        }

        if ((to_play_index % 3) == 2) {
        color = strip.Color(0, 0, 255);
        pixel =2;
        }
        
        strip.setPixelColor(pixel, color); 
        strip.show();

          midiNoteOn(0, code, 127); // play each midi code
          off[off_index] = code; // to later Off them
          off_index = off_index +1;
        }

        if (code == -2) { // end of note or chords
        // finish rendering current note or chords

        delay(duration);

         for(int i = 0; i< off_index; i++) {
           midiNoteOff(0, off[i], 127);
         }

         strip.clear();
         strip.show();
         
         off_index =0;
        } // code = -2

        if (code == -1) { // rest
        delay(duration);
        }

        } // while , scan array of midi code // render all midi codes in to_play array
        strip.clear();
        new_seed = true; // force exit and start from loop() again. get new seed 

  } // while !new_seed
    
  } // non real time


} // loop
