import numpy as np
import scipy.linalg
import os
import csv

class StateFeedbackController:
    def __init__(self, n, plant_system, ideal_system, input_signal, test_signal, sampling_rate, F_ini, F_ast):
        self.n = n
        self.m = 1
        self.r = 1
        self.Ts = 1 / sampling_rate
        self.plant_system = plant_system
        self.ideal_system = ideal_system
        self.input_signal = input_signal
        self.test_signal = test_signal
        self.F_ini = F_ini
        self.F_ast = F_ast
        print(f"Initialized StateFeedbackController class.")
    
    def save_matrices_to_csv(self, A, B, C, D, filename):
        """Save state-space matrices A, B, C, D to a CSV file."""
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Matrix', 'Values'])
            writer.writerow(['A'])
            writer.writerows(A)
            writer.writerow(['B'])
            writer.writerows(B)
            writer.writerow(['C'])
            writer.writerows(C)
            writer.writerow(['D'])
            writer.writerows(D)
        print(f"Saved matrices to {filepath}")

    def discrete_to_continuous_zoh(self, A_discrete, B_discrete, C_discrete, D_discrete, filename):
        """Convert discrete-time state-space matrices to continuous-time matrices using zero-order hold."""
        # Calculate the continuous-time A matrix
        A_cont = scipy.linalg.logm(A_discrete) / self.Ts
        
        # Calculate the continuous-time B matrix
        A_inv = np.linalg.inv(A_cont)
        B_cont = A_inv @ (A_discrete - np.eye(A_discrete.shape[0])) @ B_discrete
        
        # C and D matrices remain unchanged
        C_cont = C_discrete
        D_cont = D_discrete

        print("Converted discrete-time matrices to continuous-time using zero-order hold.")
        
        # Save the continuous-time matrices to CSV
        # self.save_matrices_to_csv(A_cont, B_cont, C_cont, D_cont, "continuous_matrices.csv")
        self.save_matrices_to_csv(A_cont, B_cont, C_cont, D_cont, filename)

        return A_cont, B_cont, C_cont, D_cont

    def analyze_system_properties(self, A_discrete, B_discrete, C_discrete, filename):
        """
        Analyze stability, controllability, and observability of the discrete-time system.
        
        Parameters:
        A_discrete : np.ndarray
            The state matrix of the discrete system.
        B_discrete : np.ndarray
            The input matrix of the discrete system.
        C_discrete : np.ndarray
            The output matrix of the discrete system.
            
        Returns:
        stable : bool
            True if the system is stable, False otherwise.
        controllable : bool
            True if the system is controllable, False otherwise.
        observable : bool
            True if the system is observable, False otherwise.
        """
        
        # Eigenvalue calculation for stability (discrete-time system)
        eigenvalues = np.linalg.eigvals(A_discrete)
        stable = np.all(np.abs(eigenvalues) < 1)  # Stability check for discrete systems
        
        # Controllability matrix calculation
        n = A_discrete.shape[0]
        controllability_matrix = np.hstack([np.linalg.matrix_power(A_discrete, i) @ B_discrete for i in range(n)])
        
        # Check controllability by matrix rank
        controllable = np.linalg.matrix_rank(controllability_matrix) == n
        
        # Observability matrix calculation
        observability_matrix = np.vstack([C_discrete @ np.linalg.matrix_power(A_discrete, i) for i in range(n)])
        
        # Check observability by matrix rank
        observable = np.linalg.matrix_rank(observability_matrix) == n

        # Optionally, save results to CSV
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # filepath = os.path.join(output_dir, 'system_properties.csv')
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Property', 'Result'])
            writer.writerow(['Stable', stable])
            writer.writerow(['Controllable', controllable])
            writer.writerow(['Observable', observable])
        
        print(f"System properties saved to {filepath}")
        
        return stable, controllable, observable

    def save_gain_to_csv(self, F_discrete):
        """Save state feedback gain to CSV file."""
        output_dir = './output'
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the path for the CSV file
        filepath = os.path.join(output_dir, "gain.csv")
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Gain Type', 'Continuous/Discrete', 'Values'])
            
            # Ensure F_discrete is an iterable (e.g., list or array)
            if isinstance(F_discrete, (np.ndarray, list)):
                # If F_discrete is a multidimensional array, flatten it
                F_discrete_values = F_discrete.flatten().tolist()
            else:
                # Convert single values into a list for compatibility with writerows
                F_discrete_values = [F_discrete]

            # Write state feedback gain (F_discrete) as a row
            writer.writerow(['State Feedback Gain (F)', 'Discrete', F_discrete_values])

        print(f"Saved gain to {filepath}")

    def calculate_and_save_discrete_eigenvalues(self, A_discrete, B_discrete, F_discrete):
        """Calculate, check stability, and save the eigenvalues of the discrete-time system."""
        
        # Reshape F_discrete to ensure correct matrix dimensions for multiplication
        F_discrete = F_discrete.reshape(1, -1)  # Ensure F_discrete is a row vector with shape (1, 4)
        
        # Closed-loop system with state feedback (discrete-time)
        A_cl_discrete = A_discrete - B_discrete @ F_discrete  # Multiplying B_discrete (4, 1) with F_discrete (1, 4)
        eigenvalues_system_discrete = np.linalg.eigvals(A_cl_discrete)

        # Check stability for the discrete-time system (all eigenvalues must have magnitudes less than 1)
        system_stable = np.all(np.abs(eigenvalues_system_discrete) < 1)

        # Save the discrete-time eigenvalues and stability result to a CSV file
        self.save_discrete_eigenvalues_and_stability_to_csv(eigenvalues_system_discrete, system_stable)

        # Output stability status for the system
        if system_stable:
            print("The system is stable.")
        else:
            print("The system is unstable.")

        print("Calculated and saved eigenvalues for the discrete-time system.")

    def save_discrete_eigenvalues_and_stability_to_csv(self, eigenvalues_system_discrete, system_stable):
        """Save the discrete-time eigenvalues of the system, along with the stability result, to a CSV file."""
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, "discrete_eigenvalues_and_stability.csv")
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Type', 'Eigenvalues', 'Stability'])

            # System eigenvalues and stability
            writer.writerow(['System Eigenvalues'])
            writer.writerow(eigenvalues_system_discrete)
            writer.writerow(['System Stability', 'Stable' if system_stable else 'Unstable'])

        print(f"Saved discrete-time eigenvalues and stability results to {filepath}")

    def simulate_with_delay_and_noise(self, system, delay_ms, noise_level, input_signal):
        """
        Simulate the output time series of the system with delay and noise, and scale the input signal
        to the range [-1, 1].
        
        Parameters:
        delay_ms (int): Time delay in milliseconds.
        noise_level (float): Amplitude of the white noise to be added (0 to 0.5).
        
        Returns:
        np.ndarray: Output time series signal with noise and delay.
        """
        # Convert delay from milliseconds to number of steps
        delay_steps = int(delay_ms / 1000 / self.Ts)
        
        # Ensure noise level is within the expected range
        noise_level = np.clip(noise_level, 0, 1e-1)
        
        # Number of time steps and input length
        num_steps = len(input_signal)
        
        # Initialize state vector (assuming zero initial state)
        state_vector = np.zeros(system.A.shape[0])
        
        # Initialize output signal array
        output_signal = np.zeros(num_steps)
        
        # Add white noise to the input signal
        noisy_input_signal = input_signal + noise_level * np.random.randn(num_steps)
        
        # Scale noisy input signal to range [-1, 1]
        """
        min_val = np.min(noisy_input_signal)
        max_val = np.max(noisy_input_signal)
        noisy_input_signal = 2 * (noisy_input_signal - min_val) / (max_val - min_val) - 1
        """
        
        # Apply delay: pad the input signal with zeros at the beginning
        delayed_input_signal = np.concatenate((np.zeros(delay_steps), noisy_input_signal))[:num_steps]
        
        # Iterate through each time step to compute the state and output
        for t in range(1, num_steps):
            # Ensure delayed_input_signal[t] is at least 1D (in case it's a scalar)
            current_input = np.atleast_1d(delayed_input_signal[t-1])
            
            # Compute the output at the current time step
            output_signal[t] = system.C @ state_vector #+ system.D @ current_input
            
            # Update the state vector for the next time step
            state_vector = system.A @ state_vector + system.B @ current_input
        
        return output_signal
    
    def simulate_without_delay_and_noise(self, system, input_signal):
        """
        Simulate the output time series of the system without delay and noise, and scale the input signal
        to the range [-1, 1].
        
        Parameters:
        input_signal (np.ndarray): The input signal to be used for the simulation.
        
        Returns:
        np.ndarray: Output time series signal.
        """
        # Number of time steps and input length
        num_steps = len(input_signal)
        
        # Initialize state vector (assuming zero initial state)
        state_vector = np.zeros(system.A.shape[0])
        
        # Initialize output signal array
        output_signal = np.zeros(num_steps)
        
        # Scale input signal to range [-1, 1]
        """
        min_val = np.min(input_signal)
        max_val = np.max(input_signal)
        scaled_input_signal = 2 * (input_signal - min_val) / (max_val - min_val) - 1
        """
        
        # Iterate through each time step to compute the state and output
        for t in range(1, num_steps):
            # Ensure scaled_input_signal[t] is at least 1D (in case it's a scalar)
            # current_input = np.atleast_1d(scaled_input_signal[t])
            current_input = np.atleast_1d(input_signal[t-1])
            
            # Compute the output at the current time step
            output_signal[t] = system.C @ state_vector #+ system.D @ current_input
            
            # Update the state vector for the next time step
            state_vector = system.A @ state_vector + system.B @ current_input
        
        return output_signal
    
    def generate_state_time_series(self, system, input_signal):
        """
        Generate a time series of state vectors of the system.
        
        Parameters:
        input_signal (np.ndarray): The input signal to be used for the simulation.
        
        Returns:
        np.ndarray: Time series of state vectors.
        """
        # Number of time steps and input length
        num_steps = len(input_signal)
        
        # Initialize state vector (assuming zero initial state)
        state_vector = np.zeros(system.A.shape[0])
        
        # Initialize time series for state vectors
        state_time_series = np.zeros((num_steps, system.A.shape[0]))
        
        # Iterate through each time step to compute the state
        for t in range(1, num_steps):
            # Ensure input_signal[t] is at least 1D (in case it's a scalar)
            current_input =  np.atleast_1d(input_signal[t-1])
            
            # Update the state vector for the next time step
            state_vector = system.A @ state_vector + system.B @ current_input
            
            # Store the current state vector in the time series
            state_time_series[t, :] = state_vector.T
        
        # print('state_time_series: ', state_time_series)
        
        return state_time_series
    
    def calculate_Gamma(self, control_input_signal, state_time_series):
        """
        Calculate the Gamma array, which represents the difference between the plant's state 
        and the ideal system's state over time.
        
        Parameters:
        - control_input_signal: The input signal for the control system, used to generate the state 
        time series of the ideal system.
        - state_time_series: A 2D array representing the state time series of the plant.
        
        Returns:
        - Gamma: A 1D array that contains the differences between the plant's state and the 
        ideal system's state over time, for each state variable in the system.
        """
        # Number of state variables (n) and time steps (N)
        n = self.n  # Number of state variables
        N = len(self.input_signal)  # Number of time steps

        # Initialize the Gamma array to store the state differences
        Gamma = np.zeros((n * N, 1))
        gamma = np.zeros((N, 1))

        # Generate the state time series for the ideal system
        generated_state_series = self.generate_state_time_series(self.ideal_system, control_input_signal)
        
        # Loop over each state variable (j represents the index of the state variable)
        for j in range(n):
            # Calculate the difference for the current state variable across all time steps
            gamma = state_time_series[:, j] - generated_state_series[:, j]

            # Store the result in the appropriate section of the Gamma array
            Gamma[j * N:(j+1) * N, 0] = gamma
            
        # Print the resulting Gamma array
        # print('Calculated Gamma matrix:', Gamma)
        
        return Gamma
    
    def generate_W(self, state_time_series):
        """
        Generate the W matrix based on system responses.

        Parameters:
        state_time_series (np.ndarray): Time series data of system states (shape: N x n, where N is the number of time steps and n is the number of states).

        Returns:
        np.ndarray: The W matrix of system responses (shape: n*N x n).
        """
        # Number of systems (n) and time steps (N)
        n = self.n  # Number of state variables
        N = len(self.input_signal)  # Number of time steps

        # Initialize the W matrix of size (n * N) x n
        W = np.zeros((n * N, n))

        # Loop over each state variable to compute the W matrix
        for i in range(n):
            # Generate the state time series for the ideal system based on the input state
            generated_state_series = self.generate_state_time_series(self.ideal_system, state_time_series[:, i])
            for j in range(n):
                # Store the generated state series for the j-th state
                W[j * N:(j + 1) * N, i] = generated_state_series[:, j]

            print('i: ', i)

        # print('Calculated W matrix: ', W)

        return W

    def compute_feedback_gain(self):
        """
        Compute the feedback gain matrix F using the formula:
        F = Gamma^T W (W^T W)^(-1)

        This function calculates the feedback gain matrix by comparing the state time series from 
        the plant system with the ideal system and performing matrix operations to derive the 
        feedback gain. The process involves generating time series data, calculating the Gamma 
        matrix and the W matrix, and computing the inverse of (W^T W) to obtain the feedback gain.

        Returns:
        F (np.ndarray): The feedback gain matrix.
        """
        
        # Generate the state time series for the plant system
        print("Generating state time series for the plant system...")
        _, control_input_signal, state_time_series = self.simulate_with_state_feedback(
            self.plant_system, self.input_signal, self.F_ini)

        # Generate the state time series for the ideal system and calculate the Gamma matrix
        print("Generating the state time series for the ideal system and calculating the Gamma matrix...")
        Gamma = self.calculate_Gamma(control_input_signal, state_time_series)

        # Compute W matrix using the state time series from the plant system
        print("Computing W matrix...")
        W = self.generate_W(state_time_series)

        # Compute the transpose of W
        print("Computing the transpose of W...")
        Wt = W.T  # Transpose of W

        # Compute W^T W
        print("Computing W^T W...")
        WtW = Wt @ W  # W^T W

        # Compute the pseudo-inverse of W^T W for better stability
        print("Computing the pseudo-inverse of W^T W...")
        WtW_inv = np.linalg.pinv(WtW)  # Pseudo-inverse provides a more stable solution when the matrix is near-singular

        # Compute the feedback gain matrix F
        print("Computing the feedback gain matrix F using the pseudo-inverse of W^T W...")
        F = Gamma.T @ W @ WtW_inv  # The feedback gain matrix

        # Save the computed feedback gain to a CSV file
        self.save_gain_to_csv(F)

        print("Feedback gain computation complete.")
        
        return F
    
    def simulate_with_state_feedback(self, system, input_signal, F, max_value=1e5):
        """
        Simulate the output time series of the system with state feedback control.
        
        Parameters:
        system: A system object with matrices A, B, C, D defining the system dynamics.
        input_signal (np.ndarray): The desired input signal (reference signal).
        F (np.ndarray): The state feedback gain matrix.
        max_value (float, optional): The maximum allowable value for control inputs and states to avoid overflow. Default is 1e5.
        
        Returns:
        tuple: (output_signal, control_input_signal) - Output time series signal and control input time series.
        """
        # Number of time steps and input length
        num_steps = len(input_signal)
        
        # Initialize state vector (assuming zero initial state)
        state_vector = np.zeros(system.A.shape[0])

        # Initialize time series for state vectors
        state_time_series = np.zeros((num_steps, system.A.shape[0]))
        
        # Initialize output signal and control input signal arrays
        output_signal = np.zeros(num_steps)
        control_input_signal = np.zeros(num_steps)
        
        # Iterate through each time step to compute the state, control input, and output
        for t in range(1, num_steps):
            # Compute the control input using state feedback
            control_input = np.atleast_1d(input_signal[t-1] - F @ state_vector)
            
            # Check for numerical overflow and clamp values if needed
            # control_input = np.clip(control_input, -max_value, max_value)
            
            # Store the control input signal
            control_input_signal[t-1] = control_input
            
            # Compute the output at the current time step
            output_signal[t] = system.C @ state_vector #+ system.D @ control_input
            
            # Update the state vector for the next time step
            state_vector = system.A @ state_vector + system.B @ control_input
            
            # Prevent state vector from exploding due to numerical instability
            # state_vector = np.clip(state_vector, -max_value, max_value)

            # Store the current state vector in the time series
            state_time_series[t, :] = state_vector.T

            # print('control_input: ', control_input)

        # print('state_time_series: ', state_time_series)

        return output_signal, control_input_signal, state_time_series

    def run(self):
        """
        Run the simulation and save both discrete and continuous matrices to CSV.

        This function performs the following tasks:
        1. Save the discrete-time matrices of both the plant and ideal systems to CSV files.
        2. Analyze and save the properties of both the plant and ideal systems to CSV files.
        3. Simulate the uncontrolled output of the plant system (without delay and noise).
        4. Compute the feedback gain (F).
        5. Calculate and save the discrete eigenvalues of the plant system with the feedback gain.
        6. Simulate the controlled output and control input signal with state feedback.
        
        Returns:
            uncontrolled_output: Output from the uncontrolled simulation.
            controlled_output: Output from the controlled simulation.
            control_input_signal: Signal used for controlling the system.
        """
        # Step 1: Save discrete-time matrices to CSV
        print("Saving discrete-time matrices for plant and ideal systems to CSV...")
        self.save_matrices_to_csv(self.plant_system.A, self.plant_system.B, self.plant_system.C, self.plant_system.D, "plant_system_discrete_matrices.csv")
        self.save_matrices_to_csv(self.ideal_system.A, self.ideal_system.B, self.ideal_system.C, self.ideal_system.D, "ideal_system_discrete_matrices.csv")

        # Step 2: Analyze and save system properties to CSV
        print("Analyzing and saving system properties for plant and ideal systems...")
        self.analyze_system_properties(self.plant_system.A, self.plant_system.B, self.plant_system.C, "plant_system_properties.csv")
        self.analyze_system_properties(self.ideal_system.A, self.ideal_system.B, self.ideal_system.C, "ideal_system_properties.csv")

        # Step 3: Simulate uncontrolled output (without delay and noise)
        print("Simulating uncontrolled output for the plant system...")
        uncontrolled_output = self.simulate_without_delay_and_noise(self.plant_system, self.test_signal)

        # Step 4: Compute feedback gain
        print("Computing feedback gain (F)...")
        F = self.compute_feedback_gain()

        # Step 5: Calculate and save discrete eigenvalues
        print("Calculating and saving discrete eigenvalues with feedback gain...")
        self.calculate_and_save_discrete_eigenvalues(self.plant_system.A, self.plant_system.B, F)

        # Step 6: Simulate controlled output with state feedback
        print("Simulating controlled output with state feedback...")
        controlled_output, control_input_signal, _ = self.simulate_with_state_feedback(self.plant_system, self.test_signal, F)

        print("Simulation complete.")
        return uncontrolled_output, controlled_output, control_input_signal

    def optimal_equalization(self):
        """
        Run the simulation and save both discrete and continuous matrices to CSV.

        This function performs the following tasks:
        1. Save the discrete-time matrices of both the plant and ideal systems to CSV files.
        2. Analyze and save the properties of both the plant and ideal systems to CSV files.
        3. Simulate the uncontrolled output of the plant system (without delay and noise).
        4. Compute the feedback gain (F).
        5. Calculate and save the discrete eigenvalues of the plant system with the feedback gain.
        6. Simulate the controlled output and control input signal with state feedback.
        
        Returns:
            uncontrolled_output: Output from the uncontrolled simulation.
            controlled_output: Output from the controlled simulation.
            control_input_signal: Signal used for controlling the system.
        """
        # Step 1: Save discrete-time matrices to CSV
        print("Saving discrete-time matrices for plant and ideal systems to CSV...")
        self.save_matrices_to_csv(self.plant_system.A, self.plant_system.B, self.plant_system.C, self.plant_system.D, "plant_system_discrete_matrices.csv")
        self.save_matrices_to_csv(self.ideal_system.A, self.ideal_system.B, self.ideal_system.C, self.ideal_system.D, "ideal_system_discrete_matrices.csv")

        # Step 2: Analyze and save system properties to CSV
        print("Analyzing and saving system properties for plant and ideal systems...")
        self.analyze_system_properties(self.plant_system.A, self.plant_system.B, self.plant_system.C, "plant_system_properties.csv")
        self.analyze_system_properties(self.ideal_system.A, self.ideal_system.B, self.ideal_system.C, "ideal_system_properties.csv")

        # Step 3: Simulate uncontrolled output (without delay and noise)
        print("Simulating uncontrolled output for the plant system...")
        uncontrolled_output = self.simulate_without_delay_and_noise(self.plant_system, self.test_signal)

        # Step 4: Compute feedback gain
        """
        print("Computing feedback gain (F)...")
        F = self.compute_feedback_gain()
        """

        # Step 5: Calculate and save discrete eigenvalues
        print("Calculating and saving discrete eigenvalues with feedback gain...")
        self.calculate_and_save_discrete_eigenvalues(self.plant_system.A, self.plant_system.B, self.F_ast)

        # Step 6: Simulate controlled output with state feedback
        print("Simulating controlled output with state feedback...")
        controlled_output, control_input_signal, _ = self.simulate_with_state_feedback(self.plant_system, self.test_signal, self.F_ast)

        print("Simulation complete.")
        return uncontrolled_output, controlled_output, control_input_signal
