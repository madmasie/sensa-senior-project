#include <math.h>
#define LOG2_NUM_SAMPLES 5      // log2 of the number of gathered samples
#define NUM_SAMPLES (1 << LOG2_NUM_SAMPLES)  // the number of gathered samples (power of two)
#define NUM_SAMPLES_M_1 (NUM_SAMPLES - 1)    // the number of gathered samples minus 1
#define SHIFT_AMOUNT (16 - LOG2_NUM_SAMPLES) // length of short (16 bits) minus log2 of number of samples

float fr[NUM_SAMPLES] ; // array of real part of samples (WINDOWED)
float fi[NUM_SAMPLES] ; // array of imaginary part of samples (WINDOWED)
float totalEnergyRatio[NUM_SAMPLES] ; // sum of harmonics magnitude up to i over total harmonic energy

void performFFT(float fr[], float fi[]) {
    // performs FFT on real sample, fr, and imaginary sample, fi
    unsigned short m;   // one of the indices being swapped
    unsigned short mr ; // the other index being swapped (r for reversed)
    fix tr, ti ; // for temporary storage while swapping, and during iteration

    int i, j ; // indices being combined in Danielson-Lanczos part of the algorithm
    int L ;    // length of the FFT's being combined
    int k ;    // used for looking up trig values from sine table

    int istep ; // length of the FFT which results from combining two FFT's

    fix wr, wi ; // trigonometric values from lookup table
    fix qr, qi ; // temporary variables used during DL part of the algorithm

    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// BIT REVERSAL //////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Bit reversal code below based on that found here: 
    // https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
    for (m=1; m<NUM_SAMPLES_M_1; m++) {
        // swap odd and even bits
        mr = ((m >> 1) & 0x5555) | ((m & 0x5555) << 1);
        // swap consecutive pairs
        mr = ((mr >> 2) & 0x3333) | ((mr & 0x3333) << 2);
        // swap nibbles ... 
        mr = ((mr >> 4) & 0x0F0F) | ((mr & 0x0F0F) << 4);
        // swap bytes
        mr = ((mr >> 8) & 0x00FF) | ((mr & 0x00FF) << 8);
        // shift down mr
        mr >>= SHIFT_AMOUNT;
        // don't swap that which has already been swapped
        if (mr<=m) continue;
        // swap the bit-reveresed indices
        tr = fr[m];
        fr[m] = fr[mr];
        fr[mr] = tr;
        ti = fi[m];
        fi[m] = fi[mr];
        fi[mr] = ti;
    }

    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// Danielson-Lanczos //////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Adapted from code by:
    // Tom Roberts 11/8/89 and Malcolm Slaney 12/15/94 malcolm@interval.com
    // Length of the FFT's being combined (starts at 1)
    L = 1;
    // Log2 of number of samples, minus 1
    k = LOG2_NUM_SAMPLES - 1;
    // While the length of the FFT's being combined is less than the number of gathered samples
    while (L < NUM_SAMPLES) {
        // Determine the length of the FFT which will result from combining two FFT's
        istep = L<<1;
        // For each element in the FFT's that are being combined . . .
        for (m=0; m<L; ++m) { 
            // Lookup the trig values for that element
            j = m << k;                         // index of the sine table
            wr =  Sinewave[j + NUM_SAMPLES/4]; // cos(2pi m/N)
            wi = -Sinewave[j];                 // sin(2pi m/N)
            wr >>= 1;                          // divide by two
            wi >>= 1;                          // divide by two
            // i gets the index of one of the FFT elements being combined
            for (i=m; i<NUM_SAMPLES; i+=istep) {
                // j gets the index of the FFT element being combined with i
                j = i + L;
                // compute the trig terms (bottom half of the above matrix)
                tr = multfix(wr, fr[j]) - multfix(wi, fi[j]);
                ti = multfix(wr, fi[j]) + multfix(wi, fr[j]);
                // divide ith index elements by two (top half of above matrix)
                qr = fr[i]>>1;
                qi = fi[i]>>1;
                // compute the new values at each index
                fr[j] = qr - tr;
                fi[j] = qi - ti;
                fr[i] = qr + tr;
                fi[i] = qi + ti;
            }
        }
        --k;
        L = istep;
    }
}

void performIFFT(float fr[], float fi[]) {
    // performs IFFT on real sample, fr, and imaginary sample, fi
    unsigned short m;   // one of the indices being swapped
    unsigned short mr ; // the other index being swapped (r for reversed)
    float tr, ti; // for temporary storage while swapping, and during iteration

    int i, j; // indices being combined in Danielson-Lanczos part of the algorithm
    int L;    // length of the FFT's being combined
    int k;    // used for looking up trig values from sine table

    int istep; // length of the FFT which results from combining two FFT's

    float wr, wi; // trigonometric values from lookup table
    float qr, qi; // temporary variables used during DL part of the algorithm

    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// BIT REVERSAL //////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Bit reversal code below based on that found here: 
    // https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
    for (m=1; m<NUM_SAMPLES_M_1; m++) {
        // swap odd and even bits
        mr = ((m >> 1) & 0x5555) | ((m & 0x5555) << 1);
        // swap consecutive pairs
        mr = ((mr >> 2) & 0x3333) | ((mr & 0x3333) << 2);
        // swap nibbles ... 
        mr = ((mr >> 4) & 0x0F0F) | ((mr & 0x0F0F) << 4);
        // swap bytes
        mr = ((mr >> 8) & 0x00FF) | ((mr & 0x00FF) << 8);
        // shift down mr
        mr >>= SHIFT_AMOUNT;
        // don't swap that which has already been swapped
        if (mr<=m) continue;
        // swap the bit-reveresed indices
        tr = fr[m];
        fr[m] = fr[mr];
        fr[mr] = tr;
        ti = fi[m];
        fi[m] = fi[mr];
        fi[mr] = ti;
    }

    //////////////////////////////////////////////////////////////////////////
    ////////////////////////// Danielson-Lanczos //////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    // Adapted from code by:
    // Tom Roberts 11/8/89 and Malcolm Slaney 12/15/94 malcolm@interval.com
    // Length of the FFT's being combined (starts at 1)
    L = 1 ;
    // Log2 of number of samples, minus 1
    k = LOG2_NUM_SAMPLES - 1;
    // While the length of the FFT's being combined is less than the number of gathered samples
    while (L < NUM_SAMPLES) {
        // Determine the length of the FFT which will result from combining two FFT's
        istep = L<<1;
        // For each element in the FFT's that are being combined . . .
        for (m=0; m<L; ++m) { 
            // Lookup the trig values for that element
            j = m << k;                         // index of the sine table
            wr =  Sinewave[j + NUM_SAMPLES/4]; // cos(2pi m/N)
            wi = Sinewave[j];                  // sin(2pi m/N)
            wr >>= 1;                          // divide by two
            wi >>= 1;                          // divide by two
            // i gets the index of one of the FFT elements being combined
            for (i=m; i<NUM_SAMPLES; i+=istep) {
                // j gets the index of the FFT element being combined with i
                j = i + L;
                // compute the trig terms (bottom half of the above matrix)
                tr = multfix(wr, fr[j]) - multfix(wi, fi[j]);
                ti = multfix(wr, fi[j]) + multfix(wi, fr[j]);
                // divide ith index elements by two (top half of above matrix)
                qr = fr[i];
                qi = fi[i];
                // compute the new values at each index
                fr[j] = qr - tr;
                fi[j] = qi - ti;
                fr[i] = qr + tr;
                fi[i] = qi + ti;
            }
        }
        --k ;
        L = istep ;
    }
}

int ratioFFT(fix fr[], fix fi[]) {
  // calculates the sum magnitude of sampled harmonics
  for (int i = 0; i < NUM_SAMPLES_M_1; i++) {
    if (i > 1) {
        totalEnergyRatio[i] += totalEnergyRatio[i-1];
    }
    totalEnergyRatio[i] += sqrt(fr[i]*fr[i]+fi[i]*fi[i]);
  }

  for (int i = 0; i < NUM_SAMPLES_M_1; i++) {
    totalEnergyRatio[i] /= totalEnergyRatio[NUM_SAMPLES_M_1];
    if (totalEnergyRatio[i] >= 0.5) {
      return i+1;
    }
  }
}

void Timer_setup_TIM2() {
   RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;       	// enable clock for TIM2
   TIM2->DIER |= (TIM_DIER_CC1IE | TIM_DIER_UIE);  // enable event gen, rcv CCR1
   TIM2->ARR = PERIOD;                         	// ARR = T = counts @4MHz
   TIM2->CCR1 = ONEMILLI;                    	// ticks for duty cycle
   TIM2->SR &= ~(TIM_SR_CC1IF | TIM_SR_UIF);   	// clr IRQ flag in status reg
   NVIC->ISER[0] |= (1 << (TIM2_IRQn & 0x1F)); 	// set NVIC interrupt: 0x1F
   __enable_irq();                             	// global IRQ enable
   TIM2->CR1 |= TIM_CR1_CEN;                   	// start TIM2 CR1
}

void TIM2_IRQHandler(void) {
   if (TIM2->SR & TIM_SR_CC1IF) {  	// triggered by CCR1 event ...
  	TIM2->SR &= ~(TIM_SR_CC1IF); 	// manage the flag
      	LED_PORT_TIMER->BSRR = (LED_PIN_1); // turn on for testing
       	TIM2->CCR1 += ONEMILLI; // increment it by one millisecond
        performIFFT(fr, fi); 	// <-- manage interrupt here
   }
   if (TIM2->SR & TIM_SR_UIF) {    	// triggered by ARR event ...
  	TIM2->SR &= ~(TIM_SR_UIF);   	// manage the flag
   }
}
