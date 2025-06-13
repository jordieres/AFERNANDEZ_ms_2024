import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
from ximu_python_library.xIMUdataClass import xIMUdataClass
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ahrs.common.orientation import q2R

# filePath = 'datasets/straightLine'
# startTime = 6
# stopTime = 26
# samplePeriod = 1/256

# filePath = 'datasets/stairsAndCorridor'
# startTime = 5
# stopTime = 53
# samplePeriod = 1/256

# filePath = 'datasets/spiralStairs'
# startTime = 4
# stopTime = 47
# samplePeriod = 1/256

filePath = r"C:\Users\Gliglo\OneDrive - Universidad Politécnica de Madrid\Documentos\UPM\TFG\Proyecto_TFG\AFERNANDEZ_ms_2024\test_InfluxDB\out\dat_2024_prueba10.xlsx"
startTime = 0
stopTime = 899.978
samplePeriod = 0.02019337


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

    # Compute absolute value
    acc_magFilt = np.abs(acc_magFilt)

    # LP filter accelerometer data
    filtCutOff = 5
    b, a = signal.butter(1, (2*filtCutOff)/(1/samplePeriod), 'lowpass')
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))


    # Threshold detection
    # A --> ORIGINAL
    # stationary = acc_magFilt < 0.01

    # B (velocidad me da negativa) --> ANGELA
    # umbral_giro = 1.0
    # stationary = (acc_magFilt < 0.01) & (np.linalg.norm([gyrX,gyrY,gyrZ],axis=0) < umbral_giro)

    # C --> ANGELA
    # # calcula la magnitud angular en cada instante
    gyro_mag = np.linalg.norm(np.vstack([gyrX, gyrY, gyrZ]).T, axis=1)
    # # umbral fijo
    # umbral_giro = 1.0  # grados/s
    # # define estacionario combinando aceleración y giro
    # stationary = (acc_magFilt < 0.01) & (gyro_mag < umbral_giro)

    # D --> ANGELA
    initPeriod = 5.0 # duración en segundos para fase inicial
    init_mask = time <= (time[0] + initPeriod)
    # fase inicial quieta 
    gyro_init = gyro_mag[init_mask]
    #umbral en función de ruido (media + 3·σ)
    umbral_giro = np.mean(gyro_init) + 3*np.std(gyro_init)
    print(f"Umbral de giro adaptativo: {umbral_giro:.2f} °/s")
    stationary = (acc_magFilt < 0.01) & (gyro_mag < umbral_giro)


    # DIBUJO DE FIGURAS
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
    ax2.set_title("accelerometer")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("acceleration (g)")
    ax2.legend(["x","y","z"])
    plt.show(block=False)

    # Compute orientation
    quat  = np.zeros((time.size, 4), dtype=np.float64)

    print("Primeros vectores aceleración rotada:")
    for i in range(3):
        acc_body = np.array([accX[i], accY[i], accZ[i]])
        acc_world = q_rot(q_conj(quat[i]), acc_body)
        print(f"t={time[i]:.2f}s: {acc_world}")
    
    # initial convergence
    initPeriod = 2
    indexSel = time<=time[0]+initPeriod
    gyr=np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
    mahony = ahrs.filters.Mahony(Kp=1, Ki=0,KpInit=1, frequency=1/samplePeriod)
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    for i in range(0, 2000):
        q = mahony.updateIMU(q, gyr=gyr, acc=acc)

    # # For all data --> codigo original
    # for t in range(0,time.size):
    #     if(stationary[t]):
    #         mahony.Kp = 0.5
    #     else:
    #         mahony.Kp = 0
    #     gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
    #     acc = np.array([accX[t],accY[t],accZ[t]])
    #     quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)

    # Modificacion angela (for all data)
    for t in range(time.size):
        mahony.Kp = 0.5 if stationary[t] else 0
        gyr_rad = np.radians([gyrX[t], gyrY[t], gyrZ[t]])
        acc_vec = np.array([accX[t], accY[t], accZ[t]])

        q_new = mahony.updateIMU(q, gyr=gyr_rad, acc=acc_vec)
        if q_new is not None:
            q = q_new  # actualiza solo si el valor es valido

        quat[t, :] = q
    print("Vectores de aceleración rotada después de fix:")
    for i in range(3):
        acc_body = np.array([accX[i], accY[i], accZ[i]])
        acc_world = q_rot(q_conj(quat[i]), acc_body)
        print(f"t={time[i]:.2f}s: {acc_world}")

    # -------------------------------------------------------------------------
    # Compute translational accelerations

    # # # Rotate body accelerations to Earth frame (COD ORIGINAL)
    # acc = []
    # for x,y,z,q in zip(accX,accY,accZ,quat):
    #     acc.append(q_rot(q_conj(q), np.array([x, y, z])))
    # acc = np.array(acc)
    # # acc = acc - np.array([0,0,1]) -->original 
    # # dos lineas siguientes= angela
    # gravity = np.array([0.0, 0.0, 1.0])
    # acc = acc - gravity
    # acc = acc * 9.81

    # modificacion angela
    acc = []
    for x, y, z, q in zip(accX, accY, accZ, quat):
        acc_body = np.array([x, y, z])
        acc_world = q_rot(q_conj(q), acc_body)  # rotar aceleración al marco global

        # notaa! --> la gravedad en el marco global (ideal) es hacia abajo
        gravity = np.array([0.0, 0.0, -1.0])  # cambia de +1 a -1
        acc_corrected = (acc_world - gravity) * 9.81
        acc.append(acc_corrected)
    acc = np.array(acc)


    # Compute translational velocities
    # acc[:,2] = acc[:,2] - 9.81

    # acc_offset = np.zeros(3)
    vel = np.zeros(acc.shape)
    for t in range(1,vel.shape[0]):
        vel[t,:] = vel[t-1,:] + acc[t,:]*samplePeriod
        if stationary[t] == True:
            vel[t,:] = np.zeros(3)

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
    plt.show(block=False)

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
    plt.show(block=False)

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
    plt.show(block=False)
    
    plt.show()

if __name__ == "__main__":
    main()