import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
from ximu_python_library.xIMUdataClass import xIMUdataClass
import numpy as np
from scipy import signal    
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.common.orientation import q2R


filePath = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\old\dat_2024_tabuenca_left.xlsx"
startTime = 0
stopTime = 359.971
samplePeriod = 0.03150455



# filePath = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\external_repos\Gait_Tracking_With_x_IMU_Python\datasets\spiralStairs_CalInertialAndMag.csv"
# startTime = 4
# stopTime = 47
# samplePeriod = 1/256

def main():
    xIMUdata = xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)
    time = xIMUdata.CalInertialAndMagneticData.Time
    gyrX = xIMUdata.CalInertialAndMagneticData.gyroscope[:,0]
    gyrY = xIMUdata.CalInertialAndMagneticData.gyroscope[:,1]
    gyrZ = xIMUdata.CalInertialAndMagneticData.gyroscope[:,2]
    accX = xIMUdata.CalInertialAndMagneticData.accelerometer[:,0]
    accY = xIMUdata.CalInertialAndMagneticData.accelerometer[:,1]
    accZ = xIMUdata.CalInertialAndMagneticData.accelerometer[:,2]

    indexSel = np.all([time>=startTime,time<=stopTime], axis=0)
    time = time[indexSel]
    gyrX = gyrX[indexSel]
    gyrY = gyrY[indexSel]
    gyrZ = gyrZ[indexSel]
    accX = accX[indexSel]
    accY = accY[indexSel]
    accZ = accZ[indexSel]


    # Compute accelerometer magnitude
    acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)

    # HP filter accelerometer data
    filtCutOff = 0.001
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'highpass')
    acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))


    print("Primeros 100 valores de acc_magFilt:")
    print(acc_magFilt[:100])

    # Compute absolute value
    acc_magFilt = np.abs(acc_magFilt)

    # LP filter accelerometer data
    filtCutOff = 5
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))


    # Threshold detection
    gyro_mag = np.linalg.norm(np.vstack([gyrX, gyrY, gyrZ]).T, axis=1)
    initPeriod = 5.0
    init_mask = time <= (time[0] + initPeriod)
    gyro_init = gyro_mag[init_mask]
    umbral_giro = np.mean(gyro_init) + 3*np.std(gyro_init)

        # En vez de acc_magFilt ya filtrado, usamos la magnitud cruda:
    acc_mag_raw = np.sqrt(accX**2 + accY**2 + accZ**2)
    acc_mag_diff = np.abs(acc_mag_raw - np.mean(acc_mag_raw[:int(1/samplePeriod)]))

    # Nuevo umbral empírico
    stationary = (acc_mag_diff < 0.015) & (gyro_mag < umbral_giro)
    print(f"Se detectaron {np.sum(stationary)} muestras estacionarias de {len(stationary)} totales.")


    # Diagnóstico de duración entre estacionarios
    durations = np.diff(np.where(np.diff(stationary.astype(int)) != 0)[0])
    print(f"Duración media entre cambios de estado estacionario: {np.mean(durations) * samplePeriod:.2f} s")



    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(time,gyrX,c='r',linewidth=0.5)
    ax1.plot(time,gyrY,c='g',linewidth=0.5)
    ax1.plot(time,gyrZ,c='b',linewidth=0.5)
    ax1.set_title("gyroscope")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("angular velocity (degrees/s)")
    ax1.legend(["x","y","z"])
    ax2.plot(time,accX,c='r',linewidth=0.5)
    ax2.plot(time,accY,c='g',linewidth=0.5)
    ax2.plot(time,accZ,c='b',linewidth=0.5)
    ax2.plot(time,acc_magFilt,c='k',linestyle=":",linewidth=1)
    ax2.plot(time,stationary,c='k')
    ax2.fill_between(time, -1, 1, where=stationary, color='gray', alpha=0.2, label='estacionario')
    ax2.legend()
    ax2.set_title("accelerometer")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("acceleration (g)")
    ax2.legend(["x","y","z"])


    # Compute orientation
    quat  = np.zeros((time.size, 4), dtype=np.float64)

    # initial convergence
    initPeriod = 2
    indexSel = time<=time[0]+initPeriod
    gyr_init = np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
    mahony = ahrs.filters.Mahony(Kp=1, Ki=0, KpInit=1, frequency=1/samplePeriod)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for i in range(2000):
        q = mahony.updateIMU(q, gyr=gyr_init, acc=acc)

    # For all data
    for t in range(time.size):
        mahony.Kp = 0.5 if stationary[t] else 0.0
        gyr = np.radians([gyrX[t], gyrY[t], gyrZ[t]])
        acc = np.array([accX[t], accY[t], accZ[t]])
        quat[t, :] = mahony.updateIMU(q, gyr=gyr, acc=acc)

    # -------------------------------------------------------------------------
    # # Compute translational accelerations

    # # Rotate body accelerations to Earth frame
    # acc = []
    # for x,y,z,q in zip(accX,accY,accZ,quat):
    #     acc.append(q_rot(q_conj(q), np.array([x, y, z])))
    # acc = np.array(acc)
    # acc = acc - np.array([0,0,1])
    # acc = acc * 9.81

    # ************************* AÑADIDO ANGELA ****************************
    acc_global = []
    for i, (x, y, z, q) in enumerate(zip(accX, accY, accZ, quat)):
        acc_body = np.array([x, y, z])
        acc_world = q_rot(q_conj(q), acc_body)

        if i < 10:  # solo mostramos los primeros 10 instantes
            print(f"t={time[i]:.2f}s → acc_world = {acc_world}")

        acc_global.append(acc_world)

    acc_global = np.array(acc_global)
    acc_global = acc_global - np.array([0, 0, 1])
    acc_global = acc_global * 9.81


    acc_global = np.array(acc_global)
    plt.figure()
    plt.plot(time, acc_global[:, 0], label='X')
    plt.plot(time, acc_global[:, 1], label='Y')
    plt.plot(time, acc_global[:, 2], label='Z')
    plt.axhline(0, color='k', linestyle='--')
    plt.title("Aceleración rotada y compensada (m/s²)")
    plt.legend()

    acc_global = np.array(acc_global)
    if t < 100:
        print(f"t={t} acc_world={acc_world}")


    # Estima bias promedio solo en fases estacionarias
    bias_est = np.mean(acc_global[stationary], axis=0)
    print(f"Bias medio estimado durante quietud: {bias_est}")

    plt.figure()
    plt.plot(time, acc_global[:,2], label='Z sin bias')
    plt.plot(time, acc_global[:,2] - bias_est[2], label='Z corregido', linestyle='--')
    plt.axhline(0, color='k', linestyle=':')
    plt.legend()
    plt.title("Comparación eje Z antes y después de bias")



    # Aplica la corrección
    acc_global -= bias_est


    alpha = 0.1  # entre 0 (reset completo) y 1 (no cambio)
    vel = np.zeros(acc_global.shape)
    for t in range(1, vel.shape[0]):
        vel[t, :] = vel[t-1, :] + acc_global[t, :] * samplePeriod
        if stationary[t]:
            vel[t, :] *= alpha# en lugar de ponerlo a cero, lo atenuamos suavemente
    # ************************* AÑADIDO ANGELA ****************************


    # # Compute translational velocities
    # # acc[:,2] = acc[:,2] - 9.81

    # # acc_offset = np.zeros(3)
    # vel = np.zeros(acc.shape)
    # for t in range(1,vel.shape[0]):
    #     vel[t,:] = vel[t-1,:] + acc[t,:]*samplePeriod
    #     if stationary[t] == True:
    #         vel[t,:] = np.zeros(3)

    # Compute integral drift during non-stationary periods
    velDrift = np.zeros(vel.shape)
    stationaryStart = np.where(np.diff(stationary.astype(int)) == -1)[0]+1
    stationaryEnd = np.where(np.diff(stationary.astype(int)) == 1)[0]+1
    for i in range(0,stationaryEnd.shape[0]):
        driftRate = vel[stationaryEnd[i]-1,:] / (stationaryEnd[i] - stationaryStart[i])
        enum = np.arange(0,stationaryEnd[i]-stationaryStart[i])
        drift = np.array([enum*driftRate[0], enum*driftRate[1], enum*driftRate[2]]).T
        velDrift[stationaryStart[i]:stationaryEnd[i],:] = drift

    # Remove integral drift
    vel = vel - velDrift
    fig = plt.figure(figsize=(10, 5))
    plt.plot(time,vel[:,0],c='r',linewidth=0.5)
    plt.plot(time,vel[:,1],c='g',linewidth=0.5)
    plt.plot(time,vel[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("velocity")
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")

    # -------------------------------------------------------------------------
    # Compute translational position
    pos = np.zeros(vel.shape)
    for t in range(1,pos.shape[0]):
        pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod

    fig = plt.figure(figsize=(10, 5))
    plt.plot(time,pos[:,0],c='r',linewidth=0.5)
    plt.plot(time,pos[:,1],c='g',linewidth=0.5)
    plt.plot(time,pos[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("position")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")


    # -------------------------------------------------------------------------
    # Plot 3D foot trajectory

    posPlot = pos
    quatPlot = quat

    extraTime = 20
    onesVector = np.ones(int(extraTime*(1/samplePeriod)))

    # Create 6 DOF animation
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object
    ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
    min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
    ax.set_zlim(min_,max_)
    ax.set_title("trajectory")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("z position (m)")

    # Gráfico XY
    plt.figure()
    plt.plot(posPlot[:,0], posPlot[:, 1])
    plt.title("Trayectoria (X vs Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid()

    print("\n--- Validación de resultados ---")

    # Rango de velocidades
    print("Rango de velocidades (m/s):")
    print("X:", np.min(vel[:, 0]), "a", np.max(vel[:, 0]))
    print("Y:", np.min(vel[:, 1]), "a", np.max(vel[:, 1]))
    print("Z:", np.min(vel[:, 2]), "a", np.max(vel[:, 2]))

    # Rango de posiciones
    print("\nRango de posiciones (m):")
    print("X:", np.min(pos[:, 0]), "a", np.max(pos[:, 0]))
    print("Y:", np.min(pos[:, 1]), "a", np.max(pos[:, 1]))
    print("Z:", np.min(pos[:, 2]), "a", np.max(pos[:, 2]))

    # Módulo de velocidad
    modulo_vel = np.linalg.norm(vel, axis=1)
    print("\nVelocidad media:", np.mean(modulo_vel), "m/s")
    print("Velocidad máxima:", np.max(modulo_vel), "m/s")

    # Duración y distancia total
    duracion_seg = time[-1] - time[0]
    distancia_total = np.linalg.norm(pos[-1])
    print("\nDuración (s):", duracion_seg)
    print("Distancia total estimada (m):", distancia_total)
    print("Velocidad media estimada:", distancia_total / duracion_seg, "m/s")
    
    plt.show()

if __name__ == "__main__":
    main()




