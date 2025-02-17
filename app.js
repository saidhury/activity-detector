let session = null;
let collecting = false;
let sensorDataBuffer = [];
let selectedFeatureIndices = null;
let selectedFeatureNames = null;

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const predictionSpan = document.getElementById('prediction');
const statusParagraph = document.getElementById('status');
const accXSpan = document.getElementById('accX');
const accYSpan = document.getElementById('accY');
const accZSpan = document.getElementById('accZ');
const gyroAlphaSpan = document.getElementById('gyroAlpha');
const gyroBetaSpan = document.getElementById('gyroBeta');
const gyroGammaSpan = document.getElementById('gyroGamma');

const windowSize = 128;
const overlap = 64;
const samplingRate = 50;

async function initialize() {
    try {
        const response = await fetch('selected_features.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        selectedFeatureIndices = new Uint32Array(data.indices);
        selectedFeatureNames = data.names;

        session = await ort.InferenceSession.create('./logistic_model.onnx');

        statusParagraph.textContent = 'Model loaded successfully. Ready to record!';
        startButton.disabled = false;

    } catch (error) {
        statusParagraph.textContent = `Error loading model: ${error}`;
        console.error(error);
    }
}
initialize();

startButton.addEventListener('click', startDataCollection);
stopButton.addEventListener('click', stopDataCollection);

async function startDataCollection() {
  collecting = true;
  startButton.disabled = true;
  stopButton.disabled = false;
  statusParagraph.textContent = "Collecting Data..."
  sensorDataBuffer = []

  if(window.DeviceMotionEvent) {
    window.addEventListener('devicemotion', handleDeviceMotion);
  } else {
    statusParagraph.textContent = 'DeviceMotionEvent API not supported.'
    stopDataCollection();
  }
}

function stopDataCollection() {
  collecting = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  statusParagraph.textContent = "Data Collection Stopped."
  window.removeEventListener('devicemotion', handleDeviceMotion);
}

function handleDeviceMotion(event) {
    if (!collecting) return;

    const acceleration = event.accelerationIncludingGravity;
    const rotationRate = event.rotationRate;

    const sensorData = [
        acceleration.x, acceleration.y, acceleration.z,
        rotationRate.alpha, rotationRate.beta, rotationRate.gamma
    ];

    accXSpan.textContent = acceleration.x ? acceleration.x.toFixed(2) : 'N/A';
    accYSpan.textContent = acceleration.y ? acceleration.y.toFixed(2) : 'N/A';
    accZSpan.textContent = acceleration.z ? acceleration.z.toFixed(2) : 'N/A';
    gyroAlphaSpan.textContent = rotationRate.alpha ? rotationRate.alpha.toFixed(2) : 'N/A';
    gyroBetaSpan.textContent = rotationRate.beta ? rotationRate.beta.toFixed(2) : 'N/A';
    gyroGammaSpan.textContent = rotationRate.gamma ? rotationRate.gamma.toFixed(2) : 'N/A';

    sensorDataBuffer.push(sensorData);

    if (sensorDataBuffer.length >= windowSize) {
        processWindow(sensorDataBuffer.slice(0, windowSize));
        sensorDataBuffer = sensorDataBuffer.slice(overlap);
    }
}

async function processWindow(windowData) {
  if (!collecting) return;
    try {
        const features = extractFeatures(windowData);

        if (features.some(isNaN) || features.length != selectedFeatureIndices.length) {
             return;
        }

        const prediction = await runInference(features);
        if (prediction !== null) {
            predictionSpan.textContent = getActivityLabel(prediction);
        }
    } catch (error) {
        console.error("Error in processWindow:", error);
    }
}

function extractFeatures(windowData) {
  const accX = [];
  const accY = [];
  const accZ = [];
  const gyroX = [];
  const gyroY = [];
  const gyroZ = [];

  for (const dataPoint of windowData) {
      accX.push(dataPoint[0] ?? 0);
      accY.push(dataPoint[1] ?? 0);
      accZ.push(dataPoint[2] ?? 0);
      gyroX.push(dataPoint[3] ?? 0);
      gyroY.push(dataPoint[4] ?? 0);
      gyroZ.push(dataPoint[5] ?? 0);
  }

  const features = [];

  for (const featureName of selectedFeatureNames) {
    let featureValue;

    if (featureName.startsWith('tBodyAcc') && !featureName.includes('Jerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(accX, accY, accZ, featureName);
    } else if (featureName.startsWith('tGravityAcc')&& !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(accX, accY, accZ, featureName);
    } else if (featureName.startsWith('tBodyAccJerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(accX, accY, accZ, featureName);
    } else if (featureName.startsWith('tBodyGyro') && !featureName.includes('Jerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(gyroX, gyroY, gyroZ, featureName);
    } else if (featureName.startsWith('tBodyGyroJerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(gyroX, gyroY, gyroZ, featureName)
    } else if (featureName.includes('Mag')) {
        if(featureName.startsWith('tBodyAccMag') || featureName.startsWith('tGravityAccMag') || featureName.startsWith('tBodyAccJerkMag')) {
            const combinedAcc = combineAxes(accX, accY, accZ);
            featureValue = calculateMagnitudeFeature(combinedAcc, featureName)
        }  else if(featureName.startsWith('tBodyGyroMag') || featureName.startsWith('tBodyGyroJerkMag') || featureName.startsWith('fBodyAccMag') || featureName.startsWith('fBodyBodyAccJerkMag')) {
            const combinedGyro = combineAxes(gyroX, gyroY, gyroZ)
            featureValue = calculateMagnitudeFeature(combinedGyro, featureName)
        } else if (featureName.startsWith('fBodyBodyGyroMag') || featureName.startsWith('fBodyBodyGyroJerkMag')) {
            const combinedGyro = combineAxes(gyroX, gyroY, gyroZ);
            featureValue = calculateMagnitudeFeature(combinedGyro, featureName);
        }
    } else if (featureName.startsWith('fBodyAcc') && !featureName.includes('Jerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(accX, accY, accZ, featureName);
    }else if (featureName.startsWith('fBodyAccJerk') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(accX, accY, accZ, featureName);
    }else if (featureName.startsWith('fBodyGyro') && !featureName.includes('Mag')) {
        featureValue = calculateSingleAxisFeature(gyroX, gyroY, gyroZ, featureName);
    }

     else if (featureName === 'angle(X,gravityMean)') {
        featureValue = angleXGravityMean(accX, accY, accZ);
    } else {
        featureValue = 0;
    }

    if (Array.isArray(featureValue)) {
        features.push(...featureValue);
    } else {
        features.push(featureValue);
    }
}
  return features;
}

function calculateSingleAxisFeature(x, y, z, featureName) {
    if (featureName.endsWith('-X')) {
        return calculateTimeDomainFeatures(x, featureName);
    } else if (featureName.endsWith('-Y')) {
        return calculateTimeDomainFeatures(y, featureName);
    } else if (featureName.endsWith('-Z')) {
        return calculateTimeDomainFeatures(z, featureName);
    }
    return 0
}

function calculateMagnitudeFeature(data, featureName) {
    return calculateTimeDomainFeatures(data, featureName)
}

function combineAxes(x, y, z) {
    const combined = [];
    for(let i = 0; i < x.length; i++){
        combined.push(Math.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]))
    }
    return combined
}

function calculateTimeDomainFeatures(data, featureName) {
    if (!data || data.length === 0) {
        return getEmptyFeatureSet(featureName);
    }

    const meanVal = mean(data);
    const stdVal = std(data, meanVal);
    const madVal = mad(data, meanVal);
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    const smaVal = sma(data);
    const energyVal = energy(data);
    const iqrVal = iqr(data);
    const entropyVal = entropy(data);
    const [arCoeff1, arCoeff2, arCoeff3, arCoeff4] = arCoeff(data);

    const featureMap = {
        'mean()': [meanVal],
        'std()': [stdVal],
        'mad()': [madVal],
        'max()': [maxVal],
        'min()': [minVal],
        'sma()': [smaVal],
        'energy()': [energyVal],
        'iqr()': [iqrVal],
        'entropy()': [entropyVal],
        'arCoeff()1': [arCoeff1],
        'arCoeff()2': [arCoeff2],
        'arCoeff()3': [arCoeff3],
        'arCoeff()4': [arCoeff4],
    };

    const baseFeatureName = featureName.split('-')[1]
    ? featureName.split('-')[1].split('()')[0] + '()'
    : featureName.split('()')[0] + '()';

    if(featureMap.hasOwnProperty(baseFeatureName)) {
        return featureMap[baseFeatureName]
    }

    const coeffRegex = /arCoeff\(\)(\d)/;
    const match = featureName.match(coeffRegex)
    if(match) {
      const coeffNum = parseInt(match[1], 10);
      return [arCoeff(data)[coeffNum-1]];
    }
    return getEmptyFeatureSet(baseFeatureName);
}

function mean(data) {
    if (!data || data.length === 0) return 0;
    return data.reduce((sum, val) => sum + val, 0) / data.length;
}

function std(data, mean) {
    if (!data || data.length === 0) return 0;
    return Math.sqrt(data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length);
}

function mad(data, meanValue) {
    if (!data || data.length === 0) return 0;
    const deviations = data.map(value => Math.abs(value - meanValue));
    return median(deviations);
}

function median(data) {
	if (!data || data.length === 0) return 0;
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function sma(data) {
    if (!data || data.length === 0) return 0;
    return data.reduce((acc, val) => acc + Math.abs(val), 0) / data.length;
}

function energy(data) {
    if (!data || data.length === 0) return 0;
    return data.reduce((acc, val) => acc + val * val, 0) / data.length;
}

function iqr(data) {
    if (!data || data.length === 0) return 0;
    const sorted = [...data].sort((a,b) => a-b)
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)]
    return q3- q1
}

function entropy(data) {
    if (!data || data.length === 0) return 0;
    const sum = data.reduce((acc, val) => acc + Math.abs(val), 0);
    const probabilities = data.map(val => Math.abs(val) / sum);
    let entropy = 0;
    for (const p of probabilities) {
        if (p > 0) {
            entropy -= p * Math.log2(p);
        }
    }
    return entropy;
}

function arCoeff(data) {
    if (!data || data.length < 4) {
        return [0, 0, 0, 0];
    }
    const r = [0, 0, 0, 0, 0];
    for (let i = 0; i < 5; i++) {
        for (let j = 0; j < data.length - i; j++) {
            r[i] += data[j] * data[j + i];
        }
    }
    const R = [
        [r[0], r[1], r[2], r[3]],
        [r[1], r[0], r[1], r[2]],
        [r[2], r[1], r[0], r[1]],
        [r[3], r[2], r[1], r[0]]
    ];
    const Rinv = numeric.inv(R);
    const b = [r[1], r[2], r[3], r[4]];
    const coefficients = numeric.dot(Rinv, b);
    return coefficients;
}

function getEmptyFeatureSet(featureName) {
    const featureMap = {
        'mean()': [0],
        'std()': [0],
        'mad()': [0],
        'max()': [0],
        'min()': [0],
        'sma()': [0],
        'energy()': [0],
        'iqr()': [0],
        'entropy()': [0],
        'arCoeff()1': [0],
        'arCoeff()2': [0],
        'arCoeff()3': [0],
        'arCoeff()4': [0],
    };

    const baseFeatureName = featureName.startsWith('arCoeff')
    ? featureName
    : featureName.split('-')[1]
        ? featureName.split('-')[1].split('()')[0] + '()'
        : featureName.split('()')[0] + '()';

    return featureMap[baseFeatureName] || [0];
}

function angleXGravityMean(accX, accY, accZ) {
    const gravity = [0,0,-1]

    if(!accX || accX.length === 0) return 0

    const meanX = mean(accX);
    const meanY = mean(accY);
    const meanZ = mean(accZ);

    const dotProduct = meanX * gravity[0] + meanY * gravity[1] + meanZ * gravity[2];
    const magnitudeA = Math.sqrt(meanX * meanX + meanY * meanY + meanZ * meanZ);
    const magnitudeB = Math.sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);

    if(magnitudeA === 0) return 0;

    const cosTheta = dotProduct / (magnitudeA * magnitudeB)
    const angleRadians = Math.acos(Math.max(-1, Math.min(1, cosTheta)));

    return angleRadians;
}

function getActivityLabel(prediction) {
    const labels = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ];
    return labels[prediction - 1];
}

async function runInference(inputData) {
    try {
        const inputTensor = new ort.Tensor('float32', inputData, [1, inputData.length]);
        const feeds = { float_input: inputTensor };
        const results = await session.run(feeds);
        const outputTensor = results['probabilities'];
        return getPrediction(outputTensor.data);

    } catch (error) {
        console.error("Inference error:", error);
        return null;
    }
}

function getPrediction(outputData) {
    if (!outputData || outputData.length === 0) {
      return 0;
    }
    let maxIndex = 0;
    let maxValue = outputData[0];
    for (let i = 1; i < outputData.length; i++) {
        if (outputData[i] > maxValue) {
            maxValue = outputData[i];
            maxIndex = i;
        }
    }
    return maxIndex + 1;
}