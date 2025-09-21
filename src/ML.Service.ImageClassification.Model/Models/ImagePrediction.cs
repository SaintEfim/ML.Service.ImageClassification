namespace ML.Service.ImageClassification.Model.Models;

public class ImagePrediction : ImageData
{
    public string PredictedLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = [];
}
