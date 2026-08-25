"""
This code aims to identify the quadrotor system ——Thrust curve,motor delay , inertia & torque coeff.
support crazyflie(cflog) and p4x(ulog).
References:Data-Driven System Identification of Quadrotors Subject to Motor Delays
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== User parameter configuration ======================
model_name = "large"        # if use P4X ulog,input large
SHOW_INSPECTION = False        #Should the flight log be displayed?
#===============The physical parameters of quadrotor============
if model_name == "crazyflie":#FLU
    MASS = 0.027
    GRAVITY = 9.81
    d = 0.028
    rotor_positions = np.array([[ d, -d, 0], [-d, -d, 0], [-d,  d, 0], [ d,  d, 0]])
    rotor_thrust_directions = np.array([[0,0,1]]*4)
    rotor_torque_directions = np.array([[0,0,-1],[0,0,1],[0,0,-1],[0,0,1]])
    log_files = [
        "D:/IngKi/graduation/quad/log10",
        "D:/IngKi/graduation/quad/log15",
        "D:/IngKi/graduation/quad/log16"
    ]
    OUTPUT_TOPIC = "motor"

elif model_name == "large":
    MASS = 3.35
    GRAVITY = 9.81
    rx, ry = 0.4179/2, 0.481332/2
    rotor_positions = np.array([[ rx, -ry, 0], [-rx,  ry, 0], [ rx,  ry, 0], [-rx, -ry, 0]])
    rotor_thrust_directions = np.array([[0,0,1]]*4)
    rotor_torque_directions = np.array([[0,0,-1],[0,0,-1],[0,0,1],[0,0,1]])
    log_files = [
        "D:/IngKi/graduation/quad/log_63_2024-1-8-16-37-54.ulg",
        "D:/IngKi/graduation/quad/log_64_2024-1-8-16-39-44.ulg",
        "D:/IngKi/graduation/quad/log_65_2024-1-8-16-40-52.ulg",
        "D:/IngKi/graduation/quad/log_66_2024-1-8-16-42-48.ulg",
    ]
    OUTPUT_TOPIC = "actuator_motors"
    from load_ulg import load_ulg

#========================Utility functions====================
def load_flight_data(file_index):
    """Load a single log and return a unified dictionary."""
    file_path = log_files[file_index]
    if model_name == "crazyflie":
        import cfusdlog
        data = cfusdlog.decode(file_path)
        ff = data["fixedFrequency"]
        ts_motor = np.array(ff["timestamp"]) / 1000.0

        motor_sp = np.column_stack([np.array(ff[f"{OUTPUT_TOPIC}.m{i+1}"]) / 65536.0 for i in range(4)])

        acc_keys = [k for k in ff.keys() if k.startswith("acc")]
        acc_map = {k[-1]: k for k in acc_keys if k[-1] in 'xyz'}
        acc_scale = {"x": 9.81, "y": 9.81, "z": 9.18}
        acc = np.column_stack([np.array(ff[acc_map[axis]]) * acc_scale[axis] for axis in "xyz"])

        gyro_keys = [k for k in ff.keys() if k.startswith("gyro")]
        gyro_map = {k[-1]: k for k in gyro_keys if k[-1] in 'xyz'}
        gyro = np.column_stack([np.array(ff[gyro_map[axis]]) / 360 * 2*np.pi for axis in "xyz"])

        #Angular acceleration: numerical gradient
        domega = np.gradient(gyro, ts_motor, axis=0, edge_order=2)
        t_acc = ts_motor.copy()
        return {
            "name": os.path.basename(file_path),
            "t_motor": ts_motor, "motor_sp": motor_sp,
            "reference_time": t_acc, "acc": acc, "gyro": gyro, "domega": domega
        }
    
    else:   # P4X
        df = load_ulg(file_path, disable_str_exceptions=True)
        
        # ---------- 1. Read accelerometer----------
        acc_cols = [f'vehicle_acceleration_xyz[{i}]' for i in range(3)]
        acc_frd = df[acc_cols].dropna()
        t_acc_orig = acc_frd.index.values.astype(float)
        acc_frd_vals = acc_frd.values

        # ---------- 2. Read gyroscope ----------
        gyro_cols = [f'vehicle_angular_velocity_xyz[{i}]' for i in range(3)]
        gyro_frd = df[gyro_cols].dropna()
        t_gyro_orig = gyro_frd.index.values.astype(float)
        gyro_frd_vals = gyro_frd.values

        # ---------- 3. Automatically select the sensor with the highest frequency as the reference time axis for interpolation. ----------
        # Calculate the median sampling interval to avoid the influence of outliers.
        dt_acc = np.median(np.diff(t_acc_orig)) if len(t_acc_orig) > 1 else 1.0
        dt_gyro = np.median(np.diff(t_gyro_orig)) if len(t_gyro_orig) > 1 else 1.0
        freq_acc = 1.0 / dt_acc
        freq_gyro = 1.0 / dt_gyro

        if freq_acc >= freq_gyro:
            base_ts = t_acc_orig
            acc_interp = acc_frd_vals         
            gyro_interp = np.zeros((len(base_ts), 3))
            for i in range(3):
                gyro_interp[:, i] = np.interp(base_ts, t_gyro_orig, gyro_frd_vals[:, i])
            gyro_final = gyro_interp
        else:
            base_ts = t_gyro_orig
            gyro_final = gyro_frd_vals        
            acc_interp = np.zeros((len(base_ts), 3))
            for i in range(3):
                acc_interp[:, i] = np.interp(base_ts, t_acc_orig, acc_frd_vals[:, i])
            acc_final  = acc_interp
        # ---------- 4. domega ----------
        domeg_cols = [f'vehicle_angular_velocity_xyz_derivative[{i}]' for i in range(3)]
        has_deriv = domeg_cols[0] in df.columns #Is angular acceleration recorded?

        if has_deriv:
            domeg_frd = df[domeg_cols].dropna()
            t_domega = domeg_frd.index.values.astype(float)
            domeg_vals = domeg_frd.values
            domega_interp = np.zeros((len(base_ts), 3))
            for i in range(3):
                domega_interp[:, i] = np.interp(base_ts, t_domega, domeg_vals[:, i])
        else:
            domega_interp = np.gradient(gyro_final, base_ts, axis=0,edge_order=2)

        # ---------- 5. Motor setpoint ----------
        motor_cols = [f'{OUTPUT_TOPIC}_control[{i}]' for i in range(4)]
        motor_raw = df[motor_cols].dropna()
        t_motor = motor_raw.index.values.astype(float)
        motor_sp = motor_raw.values

        # ---------- 6. FRD -> FLU  ----------
        acc_flu = acc_final.copy()
        acc_flu[:, 1] *= -1
        acc_flu[:, 2] *= -1

        gyro_flu = gyro_final.copy()
        gyro_flu[:, 1] *= -1
        gyro_flu[:, 2] *= -1

        domega_flu = domega_interp.copy()
        domega_flu[:, 1] *= -1
        domega_flu[:, 2] *= -1

        return {
            "name": os.path.basename(file_path),
            "t_motor": t_motor,
            "motor_sp": motor_sp,
            "reference_time": base_ts,          
            "acc": acc_flu,
            "gyro": gyro_flu,
            "domega": domega_flu
        }

def plot_flight_inspection(flights_data):
    """Visualization of motivation levels"""
    #Create a `figures` folder in the current directory to store the generated images; 
    # if the folder already exists, skip this step (`exist_ok=True`).
    os.makedirs("figures", exist_ok=True)
    for i, flight in enumerate(flights_data):
        t_m = flight["t_motor"]
        sp = flight["motor_sp"]
        thrust_exc = np.sum(sp, axis=1)
        mix = np.column_stack([rotor_positions[:,1],-rotor_positions[:,0],rotor_torque_directions[:,2]])
        exc  = sp@mix
        geo_exc_x = exc[:,0]
        geo_exc_y = exc[:,1]
        torque_z = exc[:,2]

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        # ---- Thrust ----
        axs[0].plot(t_m, thrust_exc, 'tab:green')
        axs[0].set_ylabel("Thrust exc.")
        axs[0].set_title("Linear dynamics (Z)")

        # ---- Roll & Pitch ----
        axs[1].plot(t_m, geo_exc_x, label="roll (x)")
        axs[1].plot(t_m, geo_exc_y, label="pitch (y)")
        axs[1].legend(loc='upper left', bbox_to_anchor=(1,1))
        axs[1].set_ylabel("Geo torque")
        axs[1].set_title("Angular dynamics (roll/pitch)")

        # ---- Yaw ----
        axs[2].plot(t_m, torque_z, 'tab:red')
        axs[2].set_ylabel("Torque (z)")
        axs[2].set_title("Angular dynamics (yaw)")
        axs[2].set_xlabel("Time [s]")   
        fig.suptitle(f"Flight {i}: {flight['name']} - Excitation overview", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"figures/{model_name}_flight{i}_inspection.png", dpi=150)
        plt.show()

def interactive_timeframe_selection(full_flights, purpose_str="Thrust"):
    """Interactive time segment selection"""
    print(f"\n=== Based on the stimulus graph, please select the time segment for **{purpose_str}** identification ===")
    timeframes = []
    while True:
        try:
            user_input = input(f"log_index (0~{len(full_flights)-1}),Press Enter to end the selection.: ")
            if user_input.strip() == "":
                break
            file_idx = int(user_input)
            if not (0 <= file_idx < len(full_flights)):
                print("Index out of range!")
                continue
            start = float(input("start: "))
            end   = float(input("end: "))
            if start >= end:
                print("The start second must be less than the end second.!")
                continue
            timeframes.append({"file_index": file_idx, "start": start, "end": end})
            print(f"Added segment: Log{file_idx}, {start:.1f}~{end:.1f}s")
        except ValueError:
            print("The input format is incorrect, please re-enter.")
    return timeframes

# ====================== The core functions======================
def extract_raw_data(flights_data, timeframes, fields=("acc_z",)):
    """Extract data based on time segments and return a list: (t_motor, motor_sp, t_acc, data_dict)"""
    segments = []
    for tf in timeframes:
        idx = tf["file_index"]
        flight = flights_data[idx]
        start_t, end_t = tf["start"], tf["end"]

        mask_m = (flight["t_motor"] >= start_t) & (flight["t_motor"] <= end_t)
        t_m = flight["t_motor"][mask_m]
        motor_sp = flight["motor_sp"][mask_m]

        mask_a = (flight["reference_time"] >= start_t) & (flight["reference_time"] <= end_t)
        reference_time = flight["reference_time"][mask_a]

        data_dict = {}
        if "acc_z" in fields:
            acc_z = flight["acc"][mask_a, 2].copy()
            if np.median(acc_z) < 0:
                acc_z = -acc_z
                print(f"  Automatic negation acc_z (median number {np.median(acc_z):.2f})")
            data_dict["acc_z"] = acc_z
        if "gyro" in fields:
            data_dict["gyro"] = flight["gyro"][mask_a, :].copy()
        if "domega" in fields:
            data_dict["domega"] = flight["domega"][mask_a, :].copy()

        segments.append((t_m, motor_sp, reference_time, data_dict))
    if not segments:
        raise RuntimeError("No valid data was extracted. Please check the time window configuration.")
    return segments

def filter_ema(timestamps, setpoints, Tm):
    """EMA """
    N = len(timestamps)
    rpm = np.zeros_like(setpoints)
    prev_rpm = np.zeros(setpoints.shape[1])
    prev_t = timestamps[0]
    for i in range(1,N):
        dt = timestamps[i] - prev_t
        if dt <= 0:
            rpm[i] = prev_rpm
            continue
        alpha = np.exp(-dt / Tm)
        rpm[i] = alpha * prev_rpm + (1 - alpha) * setpoints[i]
        prev_rpm = rpm[i]
        prev_t = timestamps[i]
    return rpm

def evaluate_Tm(segments_simple, Tm):
    """
    Calculate the RMSE and optimal coefficients of the thrust curve fitting for a given Tm.
    segments_simple:list: (t_motor, motor_sp, t_acc, acc_z)
    """
    all_A, all_b = [], []
    for t_m, sp, t_a, acc_z in segments_simple:
        rpm_m = filter_ema(t_m, sp, Tm)
        rpm_acc = np.zeros((len(t_a), 4)) #w_mi
        for i in range(4):
            rpm_acc[:, i] = np.interp(t_a, t_m, rpm_m[:, i])
        sum_om = np.sum(rpm_acc, axis=1)
        sum_om_sq = np.sum(rpm_acc**2, axis=1)
        A = np.column_stack([np.ones_like(sum_om)*4, sum_om, sum_om_sq])
        b = MASS * acc_z
        all_A.append(A); all_b.append(b)

    A_all = np.concatenate(all_A, axis=0)
    b_all = np.concatenate(all_b, axis=0)
    coeff, residuals, _, _ = np.linalg.lstsq(A_all, b_all, rcond=None)
    if len(residuals) > 0:
        ss_res = residuals[0]
    else:
        ss_res = np.sum((A_all @ coeff - b_all) ** 2)
    rmse = np.sqrt(ss_res / len(b_all))
    return rmse, coeff

def compute_thrust_and_torques(segments, Tm, K):
    """
    Calculate the thrust, geometric moment, and yaw torque at each moment.(Not taken K_tau)
    return dictionary: thrusts (N,4), geo_torque (N,3), yaw_torque_raw (N,)
    """
    all_thrusts, all_geo, all_yaw = [], [], []
    for t_motor, motor_sp, t_acc, _ in segments:
        rpm_motor = filter_ema(t_motor, motor_sp, Tm)
        rpm_acc = np.zeros((len(t_acc), 4))
        for i in range(4):
            rpm_acc[:, i] = np.interp(t_acc, t_motor, rpm_motor[:, i])
        f = K[0] + K[1]*rpm_acc + K[2]*rpm_acc**2   
        geo = np.zeros((len(t_acc), 3))
        yaw = np.zeros(len(t_acc))
        for i in range(4):
            F_i = np.column_stack([np.zeros_like(f[:, i]), np.zeros_like(f[:, i]), f[:, i]])
            torque_geo = np.cross(rotor_positions[i], F_i)
            geo += torque_geo
            yaw += rotor_torque_directions[i, 2] * f[:, i]
        all_thrusts.append(f)
        all_geo.append(geo)
        all_yaw.append(yaw)
    return {
        "thrusts": np.concatenate(all_thrusts, axis=0),
        "geo_torque": np.concatenate(all_geo, axis=0),#x,y torque
        "yaw_torque_raw": np.concatenate(all_yaw, axis=0)#z toeque(not taken K_tau )
    }

def estimate_inertia_and_tau(segments_rp, segments_yaw, Tm, K, inertia_ratio=1.832):
    """Identify and plot the inertia and torque coefficients."""
    # Roll/Pitch
    torques_rp = compute_thrust_and_torques(segments_rp, Tm, K)
    geo = torques_rp["geo_torque"]
    domega_rp = np.concatenate([seg[3]["domega"] for seg in segments_rp], axis=0)

    dw_x = domega_rp[:, 0]
    torque_x = geo[:, 0]
    Ixx = np.inner(dw_x, torque_x) / (dw_x**2).sum()

    dw_y = domega_rp[:, 1]
    torque_y = geo[:, 1]
    Iyy = np.inner(dw_y, torque_y) / (dw_y**2).sum()

    Izz = (Ixx + Iyy)/2 * inertia_ratio

    # Yaw
    torques_yaw = compute_thrust_and_torques(segments_yaw, Tm, K)
    yaw_raw = torques_yaw["yaw_torque_raw"]
    domega_yaw = np.concatenate([seg[3]["domega"] for seg in segments_yaw], axis=0)
    dw_z = domega_yaw[:, 2]

    K_tau = np.inner(yaw_raw, dw_z * Izz) / (yaw_raw**2).sum()

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (axis_name, I_val, torque, dw) in enumerate(
        [("x", Ixx, torque_x, dw_x), ("y", Iyy, torque_y, dw_y)]):
        ax = axs[idx]
        ax.scatter(torque, dw, s=0.5, alpha=0.3)
        x_fit = np.linspace(torque.min(), torque.max(), 100)
        ax.plot(x_fit, x_fit / I_val, 'r', label=f"$I_{{{axis_name}{axis_name}}}$ = {I_val:.3e}")
        ax.set_xlabel(f"Torque ({axis_name}) [Nm]")
        ax.set_ylabel(f"Angular acc ({axis_name}) [rad/s²]")
        ax.legend(); ax.grid(False)

    ax_z = axs[2]
    ax_z.scatter(yaw_raw, dw_z * Izz, s=0.5, alpha=0.3)
    x_fit = np.linspace(yaw_raw.min(), yaw_raw.max(), 100)
    ax_z.plot(x_fit, K_tau * x_fit, 'r', label=f"$K_\\tau$ = {K_tau:.3e}")
    ax_z.set_xlabel("Normalized yaw torque")
    ax_z.set_ylabel(f"Angular acc (z) $\\times I_{{zz}}$")
    ax_z.legend(); ax_z.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_inertia_Ktau.png", dpi=150)
    plt.show()

    return Ixx, Iyy, Izz, K_tau

# ====================== main ======================
if __name__ == "__main__":
    print(f"loading the {len(log_files)} log file of {model_name}...")
    full_flights = [load_flight_data(i) for i in range(len(log_files))]
    print("Loading complete.")

    if SHOW_INSPECTION:
        print("\nGenerating the stimulus plot; please select an appropriate time window based on the graph...")
        plot_flight_inspection(full_flights)

    # -----thrust identification-----
    timeframes_thrust = interactive_timeframe_selection(full_flights, "thrust")
    segments_thrust = extract_raw_data(full_flights, timeframes_thrust, fields=["acc_z"])

    print("\nStarting thrust identification...")
    if model_name == "crazyflie":
        Tm_range = np.linspace(0.01, 0.25, 200)
    else:
        Tm_range = np.linspace(0.01, 0.15, 200)

    # Convert to a simple tuple format.
    segs_simple = [(t_m, sp, t_a, data["acc_z"]) for t_m, sp, t_a, data in segments_thrust]

    rmse_vals = np.zeros_like(Tm_range)
    coeffs = np.zeros((len(Tm_range), 3))
    for idx, Tm in enumerate(Tm_range):
        rmse_vals[idx], coeffs[idx] = evaluate_Tm(segs_simple, Tm)

    best_idx = np.argmin(rmse_vals)
    best_Tm = Tm_range[best_idx]
    best_K = coeffs[best_idx]
    min_rmse = rmse_vals[best_idx]

    print("\n========= Thrust identification results =========")
    print(f"Motor time constant Tm = {best_Tm:.4f} s")
    print(f"Thrust coefficient K0 = {best_K[0]:.6f} N")
    print(f"Thrust coefficient K1 = {best_K[1]:.6f} N/rpm")
    print(f"Thrust coefficient K2 = {best_K[2]:.6f} N/rpm²")
    print(f"Model residuals RMSE = {min_rmse:.4f} N")

    # Thrust Plotting
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(Tm_range, rmse_vals, 'b-')
    plt.axvline(best_Tm, color='r', linestyle='--', label=f'Minimum: Tm = {best_Tm:.3f}s')
    plt.xlabel('Tm [s]'); plt.ylabel('RMSE [N]'); plt.title('Tm estimation'); plt.legend(); plt.grid(False)

    all_pred, all_true = [], []
    for t_m, sp, t_a, acc_z in segs_simple:
        rpm_m = filter_ema(t_m, sp, best_Tm)
        rpm_acc = np.array([np.interp(t_a, t_m, rpm_m[:,i]) for i in range(4)]).T
        sum_om = np.sum(rpm_acc, axis=1)
        sum_om_sq = np.sum(rpm_acc**2, axis=1)
        pred = 4*best_K[0] + best_K[1]*sum_om + best_K[2]*sum_om_sq
        all_pred.append(pred); all_true.append(MASS*acc_z)
    pred_all = np.concatenate(all_pred); true_all = np.concatenate(all_true)
    plt.subplot(1,2,2)
    plt.scatter(pred_all, true_all, s=1, alpha=0.3)
    lo, hi = min(true_all.min(), pred_all.min()), max(true_all.max(), pred_all.max())
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.xlabel('Predicted thrust [N]'); plt.ylabel('Actual thrust [N]')
    plt.title('Thrust prediction vs actual'); plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_thrust_fit.png", dpi=150)
    plt.show()
  
    # ----- Inertia identification -----
    print("\n--- Roll/Pitch Inertia Identification Time Segment ---")
    timeframes_rp = interactive_timeframe_selection(full_flights, "Roll/Pitch")
   
    print("\n--- Yaw identification time segment ---")
    timeframes_yaw = interactive_timeframe_selection(full_flights, "Yaw")

    segments_rp = extract_raw_data(full_flights, timeframes_rp, fields=["gyro", "domega"])
    segments_yaw = extract_raw_data(full_flights, timeframes_yaw, fields=["gyro", "domega"])

    Ixx, Iyy, Izz, K_tau = estimate_inertia_and_tau(segments_rp, segments_yaw, best_Tm, best_K,
                                                    inertia_ratio=1.832)
    print("\n========= Inertia/Torque Coefficient Identification Results=========")
    print(f"Ixx = {Ixx:.4e} kg·m²")
    print(f"Iyy = {Iyy:.4e} kg·m²")
    print(f"Izz = {Izz:.4e} kg·m² (Experience ratio 1.832)")
    print(f"K_tau = {K_tau:.4e} N·m/N")
    print("==========================================")

    print("\nAll identifications complete.。")

    ## 默认时间片段
#if model_name == "crazyflie":
   # default_timeframes_thrust      = [{"file_index": 0, "start": 35, "end": 65}]
    #default_timeframes_roll_pitch  = [{"file_index": 1, "start": 55, "end": 90}]
    #default_timeframes_yaw         = [{"file_index": 2, "start": 32, "end": 38},
                                     # {"file_index": 2, "start": 45, "end": 49}]
#elif model_name == "large":
    #default_timeframes_thrust      = [{"file_index": 0, "start": 10, "end": 45}]
    #default_timeframes_roll_pitch  = [{"file_index": 1, "start": 10, "end": 17},
                                     # {"file_index": 1, "start": 19, "end": 35},
                                      #{"file_index": 2, "start": 10, "end": 45}]
    #default_timeframes_yaw         = [{"file_index": 3, "start": 15, "end": 30}]