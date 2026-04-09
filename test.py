from attention_model import predict_attention

# Sample input (you can change values)
sample = {
    "pose_forward": 1,
    "phone": 0,
    "pose_x": 0.1,
    "pose_y": 0.05
}

result = predict_attention(sample)
print(result)