namespace ML.Service.ImageClassification.Model.Models;

public class ImageClassificationResponse : ImageData
{
    public string PredictedLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = [];

    public float Confidence => Score.Length > 0 ? Score.Max() : 0;
}
