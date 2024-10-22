import numpy as np
from ClassFiles.AudioLoader import AudioLoader
from ClassFiles.FrequencyResponseAnalyzer import FrequencyResponseAnalyzer
from ClassFiles.ControlSystemSimulation import ControlSystemSimulation
from ClassFiles.StateFeedbackControllerSimulation import StateFeedbackController

def main():
    # Initialize AudioLoader instance
    print("Initializing AudioLoader...")
    audio_loader = AudioLoader()

    # Define time durations for input/output and test data
    input_output_duration = 20  # Duration for input/output audio data
    test_duration = 3  # Duration for test audio data

    # Load input audio signal for a specified time period
    print("\nSelect the .wav file for the input audio signal")
    input_signal, sample_rate = audio_loader.load_audio(input_output_duration)
    print("Input audio signal loaded successfully.")

    # Load output audio signal for the same time period
    print("\nSelect the .wav file for the output audio signal")
    output_signal, _ = audio_loader.load_audio(input_output_duration)
    print("Output audio signal loaded successfully.")

    # Initialize FrequencyResponseAnalyzer and perform analysis
    print("Analyzing frequency response...")
    freq_response_analyzer = FrequencyResponseAnalyzer(
        input_signal=input_signal, 
        output_signal=output_signal, 
        sampling_rate=sample_rate, 
        time_duration=test_duration
    )
    freq_response_analyzer.analyze_and_save_bode_plot()
    print("Frequency response analysis completed and Bode plot saved.")

    # Initialize arrays to store system orders and L2 norms
    # num_iterations = 20
    num_iterations = 31
    # num_iterations = 11
    system_orders = np.zeros(num_iterations)
    l2_norms = np.zeros(num_iterations)

    for i in range(num_iterations):
        # system_order = (i + 1) * 10
        system_order = i + 145
        # system_order = i + 1
        system_orders[i] = system_order

        # Set up ControlSystemSimulation instance
        simulation = ControlSystemSimulation(
            n=system_order, 
            t_end=input_output_duration, 
            num_points=len(input_signal)
        )

        # Perform system identification using SRIM
        print(f"Identifying system for order {system_order}...")
        plant_system = simulation.identify_system_SRIM(input_signal, output_signal)

        # Simulate the system and compute L2 norm
        simulated_output = simulation.simulate_discrete_state_space(
            plant_system.A, 
            plant_system.B, 
            plant_system.C, 
            plant_system.D, 
            input_signal
        )
        l2_norms[i] = simulation.compute_fft_l2_norm(output_signal, simulated_output, sample_rate, test_duration)

        # Save results and plot
        simulation.save_results_and_plot(system_orders, l2_norms)
        print(f"System order {system_order} analysis completed.")

    # Set specific order for further control system simulation
    # selected_order = 175
    selected_order = 149
    # selected_order = 2

    # Set up ControlSystemSimulation with selected order
    print("Setting up control system simulation...")
    simulation = ControlSystemSimulation(
        n=selected_order, 
        t_end=input_output_duration, 
        num_points=len(input_signal)
    )

    # Plot input and output signals
    print("Plotting input and output signals...")
    simulation.plot_input_output(input_signal, output_signal, filename='input_output_plot.png')

    # Perform system identification using SRIM
    print("Identifying system using SRIM for selected order...")
    plant_system = simulation.identify_system_SRIM(input_signal, output_signal)
    print("System identification completed.")

    # Plot step response, eigenvalues, and Bode plot for identified system
    print("Plotting step response, eigenvalues, and Bode plot for identified system...")
    simulation.plot_step_response_SRIM(plant_system)
    simulation.plot_eigenvalues_SRIM(plant_system)
    simulation.plot_bode_SRIM(plant_system)

    # Process system matrix and save natural frequencies
    print("Saving natural frequencies from the system matrix...")
    simulation.process_matrix_and_save(plant_system.A, filename="plant_system_eigenvalues_frequencies.csv")

    # Set up State Feedback Controller
    print("Initializing State Feedback Controller...")
    state_feedback_ctrl = StateFeedbackController(
        n=selected_order, 
        plant_system=plant_system, 
        ideal_system=None, 
        input_signal=input_signal, 
        test_signal=None, 
        sampling_rate=sample_rate, 
        F_ini=None, 
        F_ast=None
    )

    # Run simulation and obtain the output signals
    print("Running State Feedback Controller simulation...")
    state_feedback_ctrl.save_matrices_to_csv(plant_system.A, plant_system.B, plant_system.C, plant_system.D, "plant_system_discrete_matrices.csv")
    state_feedback_ctrl.analyze_system_properties(plant_system.A, plant_system.B, plant_system.C, "plant_system_properties.csv")
    uncontrolled_output = state_feedback_ctrl.simulate_without_delay_and_noise(plant_system, input_signal)
    print("Simulation completed.")

    # Save the resulting audio signals
    print("Saving simulated output signals as audio files...")
    audio_loader.save_audio(uncontrolled_output, sample_rate, 'UncontrolledOutput')
    print("All audio files saved.")

if __name__ == "__main__":
    main()
