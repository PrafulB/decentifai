export const trainModel = (backend, model, trainingData, options) => {
    if (backend === 'tfjs') {
        return trainTFJSModel
    }
    if (backend === 'onnx') {
        return trainONNXModel
    }
}

const trainTFJSModel = async (model, trainingData, options = {}) => {
    const { x, y } = trainingData;
    const defaultOptions = {
        epochs: 10,
        batchSize: 32,
        verbose: 0
    };

    const trainingOptions = { ...defaultOptions, ...options };
    return await model.fit(x, y, trainingOptions);
};

const trainONNXModel = async (session, trainingData, options = {}) => {
    // TODO
};