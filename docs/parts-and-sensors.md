For the PM sensor, it seems very easy to make your own or to buy an accurate one for cheap:

<https://www.ti.com/lit/an/snla399/snla399.pdf?ts=1763783192795&ref_url=https%253A%252F%252Fwww.google.com%252F>

<https://www.analog.com/en/resources/reference-designs/circuits-from-the-lab/cn0272.html#rd-functionbenefits>

<https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/ssd/si_pd_kspd9001e.pdf>

This link shows a way in which calibration is typically done:

<https://www.ti.com/lit/ug/tidub65c/tidub65c.pdf?ts=1763772729987&ref_url=https%253A%252F%252Fwww.google.com%252F>

<https://qtwork.tudelft.nl/~schouten/linkload/phdiode.pdf>

Additionally, the ability to characterize the noise present seems very doable:

<https://www.jensign.com/transimpedance/index.html>

With whatever approach is taken, it seems necessary to characterize the noise and flaws of our measurement sensors (thus how the measured data strays from the truth can be determined before calibration).

Issues: the capacitance of the photodiode changes based on the voltage the circuit is driven with. Additionally, as in the TI datasheet, a 30 second moving average is applied to smooth noisy data. However, with the Kalman filter approach, the 30 second moving average is not necessary. While a standard Kalman filter approach cannot be used because it assumes that the distribution is gaussian, and the photons follow a Poisson distribution, the Kalman filter can be easily adapted to accept a Poisson distribution.

<https://www.strollswithmydog.com/photons-poisson-shot-noise/>

<https://econweb.ucsd.edu/cee/papers/Stohs.pdf>

<https://nipunbatra.github.io/hmm/>

<https://arxiv.org/pdf/2510.25693>