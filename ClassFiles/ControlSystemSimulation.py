import numpy as np
import scipy.linalg
import control as ctrl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import csv
from ClassFiles.SRIMAlgorithm import SRIMAlgorithm
from ClassFiles.PredictiveErrorMethod import PredictiveErrorMethod

class ControlSystemSimulation:
    def __init__(self, n, t_end=10, num_points=1000):
        self.n = n
        self.m = 1
        self.r = 1
        self.t = np.linspace(0, t_end, num_points)
        self.Ts = self.t[1] - self.t[0]
        print(f"Initialized ControlSystemSimulation class with t from 0 to {t_end} seconds and {num_points} points.")

    def generate_pwm_signal(self, frequency, duty_cycle, duration=5):
        """
        Generate a PWM signal for a given frequency, duty cycle, and duration.
        
        Parameters:
        frequency (float): The frequency of the PWM signal in Hz.
        duty_cycle (float): The duty cycle of the PWM signal as a percentage (0 to 100).
        duration (float): The duration of the PWM signal in seconds (default is 5 seconds).
        
        Returns:
        np.ndarray: Time array and the corresponding PWM signal as an array of 0s and 1s.
        """
        time_points = np.linspace(0, duration, int(duration / self.Ts))
        pwm_signal = np.zeros_like(time_points)
        period = 1 / frequency
        high_time = period * (duty_cycle / 100)
        
        for i, time in enumerate(time_points):
            if time % period < high_time:
                pwm_signal[i] = 1
            else:
                pwm_signal[i] = 0
                
        return pwm_signal
    
    def generate_exponential_swept_sine_signal(self, start_freq, end_freq, duration):
            """
            Generates an exponential swept-sine signal for the system.

            Parameters:
            start_freq (float): The starting frequency of the swept-sine signal in Hz.
            end_freq (float): The ending frequency of the swept-sine signal in Hz.
            duration (float): The duration of the signal in seconds.

            Returns:
            np.ndarray: Time array and the corresponding swept-sine signal.
            """
            # Calculate the sweep rate constant K
            K = duration / np.log(end_freq / start_freq)

            # Generate the exponential swept-sine signal
            swept_sine_signal = np.sin(2 * np.pi * start_freq * K * (np.exp(self.t / K) - 1))

            return swept_sine_signal
    
    def build_digital_system(self, A, B, C, D):
        print("Building digital system ...")
        system = ctrl.ss(A, B, C, D, self.Ts)
        print("Digital system created:", system)
        return system
    
    def simulate_discrete_state_space(self, A, B, C, D, input_signal):
        """
        Simulate the output time series of a discrete-time state-space system.

        Parameters:
        A (np.ndarray): State transition matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (np.ndarray): Feedforward matrix.
        input_signal (np.ndarray): Input time series signal.

        Returns:
        np.ndarray: Output time series signal.
        """
        # Number of time steps and input length
        num_steps = len(input_signal)
        
        # Initialize state vector (assuming zero initial state)
        state_vector = np.zeros(A.shape[0])
        
        # Initialize output signal array
        output_signal = np.zeros(num_steps)
        
        # Iterate through each time step to compute the state and output
        for t in range(num_steps):
            # Ensure input_signal[t] is at least 1D (in case it's a scalar)
            current_input = np.atleast_1d(input_signal[t])
            
            # Compute the output at the current time step
            output_signal[t] = C @ state_vector + D @ current_input
            
            # Update the state vector for the next time step
            state_vector = A @ state_vector + B @ current_input
        
        return output_signal
    
    def compute_fft_l2_norm(self, output_data, pseudo_output_signal, sampling_rate, time_test):
        """
        Compute the L2 norm of the difference between FFT results of output_data and pseudo_output_signal
        over the specified time interval using the given sampling rate.
        
        Parameters:
        - output_data: Original output signal data
        - pseudo_output_signal: Simulated output signal data
        - sampling_rate: Sampling rate for analysis
        - time_test: Time duration for data analysis

        Returns:
        - L2_norm: L2 norm of the difference between the FFT results of output_data and pseudo_output_signal
        """
        # Determine the number of samples based on time_test and sampling_rate
        num_samples = int(sampling_rate * time_test)
        
        # Extract the data within the specified interval
        output_data_slice = output_data[:num_samples]
        pseudo_output_slice = pseudo_output_signal[:num_samples]

        # Compute FFT for both slices
        fft_output_data = np.fft.fft(output_data_slice)
        fft_pseudo_output = np.fft.fft(pseudo_output_slice)

        # Calculate the difference and compute the L2 norm
        fft_difference = fft_output_data - fft_pseudo_output
        l2_norm = np.linalg.norm(fft_difference, ord=2)

        return l2_norm

    def save_results_and_plot(self, system_orders, l2_norms):
        """
        Save the system orders and L2 norms to a CSV file and create a plot of the results.
        
        Parameters:
        - system_orders: List of system orders
        - l2_norms: List of corresponding L2 norms

        Outputs:
        - A CSV file and a PNG plot saved in the './output' directory.
        """
        # Ensure the output directory exists
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the results to a CSV file
        csv_file_path = os.path.join(output_dir, 'results.csv')
        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['System Order', 'L2 Norm'])
            for order, norm in zip(system_orders, l2_norms):
                csv_writer.writerow([order, norm])
        
        print(f"Results saved to {csv_file_path}")

        # Plot the results
        plt.figure()
        plt.plot(system_orders, l2_norms, marker='o', linestyle='-')
        plt.xlabel('System Order')
        plt.ylabel('L2 Norm')
        plt.title('L2 Norm vs System Order')
        plt.grid(True)

        # Save the plot as a PNG file
        plot_file_path = os.path.join(output_dir, 'results_plot.png')
        plt.savefig(plot_file_path)
        plt.close()

        print(f"Plot saved to {plot_file_path}")

    def impulse_response(self, system):
        """
        Compute and save the impulse response of a given system.

        Parameters:
        system: The system object to compute the impulse response for.
        """
        print("Computing impulse response for the given system.")
        
        # Compute impulse response
        T, yout = ctrl.impulse_response(system, T=self.t)
        print("Impulse response computed.")
        
        # Ensure the output directory exists
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot and save the impulse response
        plt.figure()
        plt.plot(T, yout, label='Impulse Response')
        plt.title('Impulse Response of the System')
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        file_path = os.path.join(output_dir, 'impulse_response.png')
        plt.savefig(file_path)
        plt.close()
        
        print(f"Impulse response graph saved as '{file_path}'.")

    def plot_input_output(self, input_signal, output_signal, filename='input_output_plot.png'):
        """
        Plots time series data of input and output signals on the same graph and saves it as a PNG.

        Args:
            input_signal (np.array): Time series data of the input signal.
            output_signal (np.array): Time series data of the output signal.
        """
        # Ensure the output directory exists
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))

        # Plot input signal
        plt.plot(self.t, input_signal, label='Input Signal', linewidth=3.0)

        # Plot output signal
        plt.plot(self.t, output_signal, label='Output Signal', linewidth=2.0)

        # Graph settings
        plt.title('Input and Output Signal Over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Save as PNG
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory

        print(f"Plot saved as {file_path}")

    def identify_system_SRIM(self, input_signals, output_signals):
        """
        Identify the system using the SRIM Algorithm (State-space Realization by Input-Output Matrix) method.
        
        Parameters:
        input_signals (array): The input signals to the system.
        output_signals (array): The output signals from the system.

        Returns:
        identified_system: The identified continuous-time system model.
        """
        print("Identifying system using SRIM Algorithm ...")
        num_samples = len(input_signals)

        # Instantiate SRIM algorithm with necessary parameters
        srim_algorithm = SRIMAlgorithm(n=self.n, 
                                    m=self.m, 
                                    r=self.r, 
                                    l=num_samples, 
                                    Y=output_signals, 
                                    U=input_signals, 
                                    mode=0)

        # Run the SRIM algorithm to identify system matrices
        A_matrix, B_matrix, C_matrix, D_matrix = srim_algorithm.run()

        # Construct the identified system as a continuous-time system model
        identified_system = self.build_digital_system(A_matrix, B_matrix, C_matrix, D_matrix)

        return identified_system

    def identify_system_PEM(self, input_signals, output_signals):
        """
        Identify the system using the Predictive Error Method (PEM).
        
        Parameters:
        input_signals (array): The input signals to the system.
        output_signals (array): The output signals from the system.

        Returns:
        identified_system: The identified continuous-time system model.
        """
        print("Identifying system using Predictive Error Method ...")

        # Instantiate the PredictiveErrorMethod class for system identification
        pem_algorithm = PredictiveErrorMethod(input_data=input_signals, 
                                            output_data=output_signals, 
                                            system_order=self.n)

        # Estimate system matrices using PEM
        A_matrix, B_matrix, C_matrix, D_matrix = pem_algorithm.identify_state_space()

        # Construct the identified system as a continuous-time system model
        identified_system = self.build_digital_system(A_matrix, B_matrix, C_matrix, D_matrix)

        return identified_system
    

    def plot_step_response(self, original_system, identified_system):
        print("Plotting step response for original and identified systems ...")
        T, yout_original = ctrl.step_response(original_system, T=self.t, input=0)
        T, yout_identified = ctrl.step_response(identified_system, T=self.t, input=0)

        plt.figure()
        plt.plot(T, yout_original, label='Original System', linewidth=3.0)
        plt.plot(T, yout_identified, label='Identified System', linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.legend()
        plt.title('Step Response Comparison')

        file_path = './output/step_response_comparison.png'
        plt.savefig(file_path)
        print(f"Step response plot saved as '{file_path}'.")

    def plot_eigenvalues(self, original_system, identified_system):
        print("Plotting eigenvalues for original and identified systems ...")
        A_original = scipy.linalg.logm(original_system.A) / self.Ts
        A_identified = scipy.linalg.logm(identified_system.A) / self.Ts
        eig_original = np.linalg.eigvals(A_original)
        eig_identified = np.linalg.eigvals(A_identified)

        plt.figure()
        plt.scatter(np.real(eig_original), np.imag(eig_original), label='Original System', marker='o')
        plt.scatter(np.real(eig_identified), np.imag(eig_identified), label='Identified System', marker='x')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Plot')
        plt.xlim(None, 0)
        plt.legend()
        plt.grid(True)

        file_path = './output/eigenvalue_comparison.png'
        plt.savefig(file_path)
        print(f"Eigenvalue plot saved as '{file_path}'.")

    def plot_bode(self, original_system, identified_system):
        print("Plotting Bode plot for original and identified systems ...")
        plt.figure()

        # Bode plot for the original and identified systems
        mag_original, phase_original, omega_original = ctrl.bode(original_system, plot=False)
        mag_identified, phase_identified, omega_identified = ctrl.bode(identified_system, plot=False)

        # Convert omega (rad/s) to frequency in Hz for both systems
        freq_hz_original = omega_original / (2 * np.pi)
        freq_hz_identified = omega_identified / (2 * np.pi)

        # Magnitude plot
        plt.subplot(2, 1, 1)
        plt.semilogx(freq_hz_original, 20 * np.log10(mag_original), label='Original System', linewidth=3.0)
        plt.semilogx(freq_hz_identified, 20 * np.log10(mag_identified), label='Identified System', linewidth=2.0)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude [dB]')
        plt.legend()

        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(freq_hz_original, phase_original, label='Original System', linewidth=3.0)
        plt.semilogx(freq_hz_identified, phase_identified, label='Identified System', linewidth=2.0)
        plt.ylabel('Phase [deg]')
        plt.xlabel('Frequency [Hz]')
        plt.legend()

        # Save the plot
        file_path = './output/bode_plot_comparison.png'
        plt.savefig(file_path)
        print(f"Bode plot saved as '{file_path}'.")
    
    
    def plot_step_response_PlantVsIdeal(self, Plant_system, Ideal_system):
        """Plot step response comparison between Plant and Ideal systems."""
        print("Plotting step response for Plant and Ideal systems...")
        
        # Calculate step response for both systems
        time, response_plant = ctrl.step_response(Plant_system, T=self.t[0:800], input=0)
        time, response_ideal = ctrl.step_response(Ideal_system, T=self.t[0:800], input=0)

        # Plot the step responses
        plt.figure()
        plt.plot(time, response_plant, label='Plant System', linewidth=3.0)
        plt.plot(time, response_ideal, label='Ideal System', linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.legend()
        plt.title('Step Response Comparison (Plant vs Ideal)')

        # Save the plot
        file_path = './output/step_response_comparison_PlantVsIdeal.png'
        plt.savefig(file_path)
        print(f"Step response plot saved as '{file_path}'.")

    def plot_eigenvalues_PlantVsIdeal(self, Plant_system, Ideal_system):
        """Plot eigenvalue comparison between Plant and Ideal systems."""
        print("Plotting eigenvalues for Plant and Ideal systems...")

        # Calculate discrete-time A matrices and eigenvalues for both systems
        A_Plant = scipy.linalg.logm(Plant_system.A) / self.Ts
        A_Ideal = scipy.linalg.logm(Ideal_system.A) / self.Ts
        eig_Plant = np.linalg.eigvals(A_Plant)
        eig_Ideal = np.linalg.eigvals(A_Ideal)

        # Plot the eigenvalues
        plt.figure()
        plt.scatter(np.real(eig_Plant), np.imag(eig_Plant), label='Plant System', marker='o')
        plt.scatter(np.real(eig_Ideal), np.imag(eig_Ideal), label='Ideal System', marker='x')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Plot (Plant vs Ideal)')
        plt.xlim(None, 0)
        plt.legend()
        plt.grid(True)

        # Save the plot
        file_path = './output/eigenvalue_comparison_PlantVsIdeal.png'
        plt.savefig(file_path)
        print(f"Eigenvalue plot saved as '{file_path}'.")

    def plot_bode_PlantVsIdeal(self, Plant_system, Ideal_system):
        """Plot Bode comparison between Plant and Ideal systems."""
        print("Plotting Bode plot for Plant and Ideal systems...")
        plt.figure(figsize=(12, 8))

        # Calculate Bode plot for both systems
        mag_plant, phase_plant, omega_plant = ctrl.bode(Plant_system, plot=False)
        mag_ideal, phase_ideal, omega_ideal = ctrl.bode(Ideal_system, plot=False)

        # Convert angular frequency (rad/s) to frequency in Hz
        freq_hz_plant = omega_plant / (2 * np.pi)
        freq_hz_ideal = omega_ideal / (2 * np.pi)

        # Custom formatter for frequency axis
        def custom_formatter(x, pos):
            return '{:.0f}k'.format(x / 1000) if x >= 1000 else '{:.0f}'.format(x)

        # Magnitude plot
        plt.subplot(2, 1, 1)
        plt.semilogx(freq_hz_plant, 20 * np.log10(mag_plant), label='Plant System', linewidth=3.0)
        plt.semilogx(freq_hz_ideal, 20 * np.log10(mag_ideal), label='Ideal System', linewidth=2.0)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude [dB]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(freq_hz_plant, phase_plant, label='Plant System', linewidth=3.0)
        plt.semilogx(freq_hz_ideal, phase_ideal, label='Ideal System', linewidth=2.0)
        plt.ylabel('Phase [deg]')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()

        # Save the plot
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 'bode_plot_comparison_PlantVsIdeal.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Bode plot saved as '{file_path}'.")
    

    def plot_step_response_SRIMvsPEM(self, SRIM_system, PEM_system):
        print("Plotting step response for SRIM and PEM ...")
        T, yout_original = ctrl.step_response(SRIM_system, T=self.t[0:800], input=0)
        T, yout_identified = ctrl.step_response(PEM_system, T=self.t[0:800], input=0)

        plt.figure()
        plt.plot(T, yout_original, label='Identified System using SRIM', linewidth=3.0)
        plt.plot(T, yout_identified, label='Identified System using PEM', linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.legend()
        plt.title('Step Response Comparison (SRIM vs PEM)')

        file_path = './output/step_response_comparison_SRIMvsPEM.png'
        plt.savefig(file_path)
        print(f"Step response plot saved as '{file_path}'.")

    def plot_eigenvalues_SRIMvsPEM(self, SRIM_system, PEM_system):
        print("Plotting eigenvalues for SRIM and PEM ...")
        A_SRIM = scipy.linalg.logm(SRIM_system.A) / self.Ts
        A_PEM = scipy.linalg.logm(PEM_system.A) / self.Ts
        eig_SRIM = np.linalg.eigvals(A_SRIM)
        eig_PEM = np.linalg.eigvals(A_PEM)

        plt.figure()
        plt.scatter(np.real(eig_SRIM), np.imag(eig_SRIM), label='Identified System using SRIM', marker='o')
        plt.scatter(np.real(eig_PEM), np.imag(eig_PEM), label='Identified System using PEM', marker='x')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Plot (SRIM vs PEM)')
        plt.xlim(None, 0)
        plt.legend()
        plt.grid(True)

        file_path = './output/eigenvalue_comparison_SRIMvsPEM.png'
        plt.savefig(file_path)
        print(f"Eigenvalue plot saved as '{file_path}'.")

    def plot_bode_SRIMvsPEM(self, SRIM_system, PEM_system):
        print("Plotting Bode plot for SRIM and PEM ...")
        plt.figure(figsize=(12, 8))

        # Bode plot for the SRIM and PEM systems
        mag_srim, phase_srim, omega_srim = ctrl.bode(SRIM_system, plot=False)
        mag_pem, phase_pem, omega_pem = ctrl.bode(PEM_system, plot=False)

        # Convert omega (rad/s) to frequency in Hz for both systems
        freq_hz_srim = omega_srim / (2 * np.pi)
        freq_hz_pem = omega_pem / (2 * np.pi)

        # Custom formatter to display frequency in 'kHz' for values >= 1000 Hz
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # Magnitude plot
        plt.subplot(2, 1, 1)
        plt.semilogx(freq_hz_srim, 20 * np.log10(mag_srim), label='Identified System using SRIM', linewidth=3.0)
        plt.semilogx(freq_hz_pem, 20 * np.log10(mag_pem), label='Identified System using PEM', linewidth=2.0)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude [dB]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')

        # Set fewer major ticks on the x-axis (frequency axis)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(freq_hz_srim, phase_srim, label='Identified System using SRIM', linewidth=3.0)
        plt.semilogx(freq_hz_pem, phase_pem, label='Identified System using PEM', linewidth=2.0)
        plt.ylabel('Phase [deg]')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')

        # Set fewer major ticks on the x-axis (frequency axis)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()

        # Save the plot
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 'bode_plot_comparison_SRIMvsPEM.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Bode plot saved as '{file_path}'.")
    

    def plot_step_response_SRIM(self, SRIM_system):
        print("Plotting step response for SRIM ...")
        T, yout_original = ctrl.step_response(SRIM_system, T=self.t[0:800], input=0)

        plt.figure()
        plt.plot(T, yout_original, label='Identified System using SRIM', linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Response')
        plt.legend()
        plt.title('Step Response (SRIM)')

        file_path = './output/step_response_SRIM.png'
        plt.savefig(file_path)
        print(f"Step response plot saved as '{file_path}'.")

    def plot_eigenvalues_SRIM(self, SRIM_system):
        print("Plotting eigenvalues for SRIM ...")
        A_SRIM = scipy.linalg.logm(SRIM_system.A) / self.Ts
        eig_SRIM = np.linalg.eigvals(A_SRIM)

        plt.figure()
        plt.scatter(np.real(eig_SRIM), np.imag(eig_SRIM), label='Identified System using SRIM', marker='x')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Plot (SRIM)')
        plt.legend()
        plt.grid(True)

        file_path = './output/eigenvalue_SRIM.png'
        plt.savefig(file_path)
        print(f"Eigenvalue plot saved as '{file_path}'.")

    def plot_bode_SRIM(self, SRIM_system):
        print("Plotting Bode plot for SRIM ...")
        plt.figure(figsize=(12, 8))

        # Bode plot for the SRIM system
        mag_srim, phase_srim, omega_srim = ctrl.bode(SRIM_system, plot=False)

        # Convert omega (rad/s) to frequency in Hz
        freq_hz_srim = omega_srim / (2 * np.pi)

        # Custom formatter to display frequency in 'kHz' for values >= 1000 Hz
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # Magnitude plot
        plt.subplot(2, 1, 1)
        plt.semilogx(freq_hz_srim, 20 * np.log10(mag_srim), label='Identified System using SRIM', linewidth=2.0)
        plt.title('Bode Plot')
        plt.ylabel('Magnitude [dB]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')

        # Set fewer major ticks on the x-axis (frequency axis)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(freq_hz_srim, phase_srim, label='Identified System using SRIM', linewidth=2.0)
        plt.ylabel('Phase [deg]')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.grid(which='both', linestyle='-', linewidth='0.5')

        # Set fewer major ticks on the x-axis (frequency axis)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=6))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()

        # Save the plot
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 'bode_plot_SRIM.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Bode plot saved as '{file_path}'.")
    

    # Function to compute matrix logarithm and scale by Ts
    def compute_scaled_logarithm(self, A):
        """
        Compute the matrix logarithm of A and scale by the sampling time Ts.
        """
        A_log = scipy.linalg.logm(A) / self.Ts
        return A_log

    # Function to compute eigenvalues of a matrix
    def compute_eigenvalues(self, A):
        """
        Compute the eigenvalues of matrix A.
        """
        eigvals = np.linalg.eigvals(A)
        return eigvals

    # Function to compute natural frequencies (Hz) from the imaginary parts of the eigenvalues
    def compute_natural_frequencies(self, eigvals):
        """
        Compute natural frequencies (in Hz) from the imaginary part of eigenvalues.
        """
        eig_freqs = np.abs(np.imag(eigvals)) / (2 * np.pi)
        return eig_freqs

    # Function to sort eigenvalues based on real part in descending order
    def sort_eigenvalues_by_real_part(self, eigvals):
        """
        Sort the eigenvalues by their real part in descending order.
        """
        eigvals_sorted = sorted(eigvals, key=lambda x: x.real, reverse=True)
        return eigvals_sorted

    # Function to save real parts of eigenvalues and natural frequencies to a CSV file
    def save_to_csv(self, eigvals, frequencies, filename="eigenvalues_frequencies.csv"):
        """
        Save eigenvalues' real parts and corresponding natural frequencies to a CSV file in the ./output directory,
        ensuring no duplicate rows are written and sorted by the real part in descending order.
        """
        # Create output directory if it doesn't exist
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use a set to track unique (real part, frequency) pairs
        unique_data = set()
        
        # Collect unique data
        for eig, freq in zip(eigvals, frequencies):
            if freq > 0:  # Only save conjugate pairs with positive frequencies
                unique_data.add((eig.real, freq))

        # Sort data by the real part in descending order
        sorted_data = sorted(unique_data, key=lambda x: x[0], reverse=True)

        # Save the CSV file in the output directory
        file_path = os.path.join(output_dir, filename)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Real Part", "Frequency (Hz)"])
            
            # Write sorted data to CSV
            for real_part, freq in sorted_data:
                writer.writerow([real_part, freq])

    # Function that combines all steps
    def process_matrix_and_save(self, A, filename="eigenvalues_frequencies.csv"):
        """
        Full process: 
        1. Compute matrix logarithm scaled by Ts
        2. Compute eigenvalues
        3. Sort eigenvalues by real part
        4. Compute natural frequencies from the eigenvalues' imaginary parts
        5. Save the real parts and natural frequencies to a CSV file
        """
        # Step 1: Compute scaled logarithm of A
        A_log = self.compute_scaled_logarithm(A)
        
        # Step 2: Compute eigenvalues
        eigvals = self.compute_eigenvalues(A_log)
        
        # Step 3: Sort eigenvalues by real part
        eigvals_sorted = self.sort_eigenvalues_by_real_part(eigvals)
        
        # Step 4: Compute natural frequencies from imaginary parts
        frequencies = self.compute_natural_frequencies(eigvals_sorted)
        
        # Step 5: Save results to CSV
        self.save_to_csv(eigvals_sorted, frequencies, filename)
