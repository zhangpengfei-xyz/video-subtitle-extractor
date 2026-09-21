"""CPU inference adapters; preserve PaddleOCR's preprocessing and decoding."""
from pathlib import Path


class OpenVINOInfer:
    def __init__(self, path, threads):
        import openvino as ov

        self.model = ov.Core().compile_model(str(path), 'CPU', {
            'INFERENCE_NUM_THREADS': threads,
            'NUM_STREAMS': 1,
            'PERFORMANCE_HINT': 'LATENCY',
            'INFERENCE_PRECISION_HINT': ov.Type.f32,
        })
        self.request = self.model.create_infer_request()

    def __call__(self, x):
        result = self.request.infer(dict(enumerate(x)), share_inputs=True)
        return [result[output] for output in self.model.outputs]


class ONNXRuntimeInfer:
    def __init__(self, path, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.inputs = [item.name for item in self.session.get_inputs()]

    def __call__(self, x):
        return self.session.run(None, dict(zip(self.inputs, x)))


def use_cpu_backend(recogniser, model_config, backend, threads):
    adapter = {'openvino': OpenVINOInfer, 'onnxruntime': ONNXRuntimeInfer}[backend]
    pipeline = recogniser.paddlex_pipeline._pipeline
    for model, directory in zip((pipeline.text_det_model, pipeline.text_rec_model),
                                (model_config.DET_MODEL_PATH, model_config.REC_MODEL_PATH)):
        model.infer = adapter(Path(directory) / 'inference.onnx', threads)
