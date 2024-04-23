import onnx
import numpy as np
import onnxruntime as ort

def checker(path, model_outputs, input):

    # Path of the model
    onnx_model_path = path

    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model, 
                            full_check= True,
                            skip_opset_compatibility_check= False,
                            check_custom_domain= True)

    # Check one
    onnx.checker.check_model(onnx_model_path, 
                            full_check= True,
                            skip_opset_compatibility_check= False,
                            check_custom_domain= True)

    # Load the ONNX model
    session = ort.InferenceSession(onnx_model_path)

    # Get the input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()

    # Create input data
    input_data = input.detach().numpy()

    # Run inference
    outputs = session.run([i.name for i in output_name], {input_name: input_data})

    # Printing all output's shape
    print('Output shape in order are given below:')
    model_outputs = list(model_outputs[0].values())
    for converted_output, original_output in zip(outputs, model_outputs):
        if converted_output.shape == original_output.shape:
            print('Shape matches.')
            print(converted_output.shape)
            print(original_output.shape)
            if np.allclose(converted_output, original_output.detach().numpy(), rtol=0.1, atol=0.1):
                print('Output overall also matches :)')
            
            else:
                print("But output values do not match :(")
                print('Start')
                print(original_output.detach().numpy())
                print('\n---00000000000000000000000000000000000000---\n')
                print(converted_output)
                print('End')
    