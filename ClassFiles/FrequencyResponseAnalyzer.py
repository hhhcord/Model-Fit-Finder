import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy.signal import welch, csd, butter, lfilter

class FrequencyResponseAnalyzer:
    def __init__(self, input_signal, output_signal, sampling_rate, time_duration):
        self.input_signal = input_signal  # Input signal
        self.output_signal = output_signal  # Output signal
        self.sampling_rate = sampling_rate  # Sampling rate
        self.cutoff_frequency = 5e2  # Cutoff frequency (Hz)
        self.time_duration = time_duration  # Duration of the signal

    def apply_low_pass_filter(self, signal, filter_sampling_rate=2000):
        # Apply low-pass filter
        nyquist = 0.5 * filter_sampling_rate
        normal_cutoff = self.cutoff_frequency / nyquist
        b, a = butter(N=1, Wn=normal_cutoff, btype='low', analog=False)
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def compute_power_spectrum(self):
        # Calculate power spectrum
        frequencies, power_spectrum = welch(self.input_signal, fs=self.sampling_rate)
        return frequencies, power_spectrum

    def compute_cross_spectrum(self):
        # Calculate cross spectrum
        frequencies, cross_spectrum = csd(self.input_signal, self.output_signal, fs=self.sampling_rate)
        return frequencies, cross_spectrum

    def compute_frequency_response(self):
        # Calculate frequency response function
        frequencies, power_spectrum = self.compute_power_spectrum()
        _, cross_spectrum = self.compute_cross_spectrum()
        frequency_response = cross_spectrum / power_spectrum
        return frequencies, frequency_response

    def save_bode_plot(self, frequencies, frequency_response, output_dir="./output"):
        # Save Bode plot (gain and phase)
        gain = 20 * np.log10(np.abs(frequency_response))  # Gain (in decibels)
        phase = np.angle(frequency_response, deg=True)  # Phase (in degrees)

        gain = self.apply_low_pass_filter(gain)
        phase = self.apply_low_pass_filter(phase)

        # Create directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Custom formatter
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # Gain plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.semilogx(frequencies, gain, base=2)
        plt.title('Bode Plot')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(frequencies, phase, base=2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [degrees]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(output_dir, 'bode_plot.png')
        plt.savefig(output_file)
        plt.close()

    def analyze_and_save_bode_plot(self):
        # Analyze frequency response using input and output data, and save the Bode plot
        frequencies, frequency_response = self.compute_frequency_response()
        self.save_bode_plot(frequencies, frequency_response)
