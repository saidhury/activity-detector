import joblib
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx._parse import _parse_sklearn_classifier
import onnx
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnxruntime import InferenceSession
import json
from sklearn.linear_model import LogisticRegression


# Custom Parser and Shape Calculator 
def skl2onnx_parse_classifier(scope, model, inputs, custom_parsers=None):
    alias = 'Sklearn' + model.__class__.__name__
    op = scope.declare_local_operator(alias, model)
    op.inputs = inputs
    return _parse_sklearn_classifier(scope, model, inputs)

def skl2onnx_shape_calculator(operator):
  op = operator.raw_operator
  if (type(op) == LogisticRegression):
        calculate_linear_classifier_output_shapes(operator)
  else:
    raise RuntimeError("This script is for parsing logistic regression only, got %r." % type(op))

# Load the Trained Model
model = joblib.load('trained_logistic_model.joblib')
with open('selected_features.json', 'r') as f:
    feature_info = json.load(f)
    selected_feature_indices = feature_info['indices']
num_features = len(selected_feature_indices)
print(f"Number of selected features: {num_features}")

# Defining Input Type
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# Converting to ONNX 
try:
    update_registered_converter(
        LogisticRegression, 'SklearnLinearClassifier',
        skl2onnx_shape_calculator,
        skl2onnx_parse_classifier,
        # options={'zipmap': False, 'output_class_labels': False} # For older versions
    )
except:
    pass

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12,  # Use a specific opset version (important!)
)

# Creating a New ONNX Model with Explicit Outputs 
original_graph = onnx_model.graph

# Define output information:  A single float tensor for probabilities.
output_tensor = make_tensor_value_info(
    name='probabilities',  # Name the output tensor
    elem_type=onnx.TensorProto.FLOAT,
    shape=[None, len(model.classes_)]  # [batch_size, num_classes]
)
# Create a new graph with the original nodes + explicit output.
new_graph = make_graph(
    nodes=original_graph.node,  # All the original nodes
    name='LogisticRegressionGraph',
    inputs=original_graph.input,
    outputs=[output_tensor],  # Our defined output
    initializer=original_graph.initializer, # copy weights
)

# Create the new ONNX model.
new_model = make_model(new_graph)
new_model.opset_import[0].version = 12 # make sure opset version is consistent

# Save the ONNX Model
with open("logistic_model.onnx", "wb") as f:
    f.write(new_model.SerializeToString())

print("ONNX model saved as logistic_model.onnx")

# Verify the ONNX Model
try:
    onnx.checker.check_model(new_model)
    print("ONNX model check passed.")
    # Also, verify with onnxruntime:
    sess = InferenceSession(new_model.SerializeToString())
    print("ONNX model loaded successfully with onnxruntime.")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Input name: {input_name}, Output name: {output_name}")


except onnx.checker.ValidationError as e:
    print(f"ONNX model check failed: {e}")
except Exception as e:
    print(f"Error loading or verifying with onnxruntime: {e}")