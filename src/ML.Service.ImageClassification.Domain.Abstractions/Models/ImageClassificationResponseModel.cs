namespace ML.Service.ImageClassification.Domain.Models;

public class ImageClassificationResponseModel
{
    public string PredictedLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = [];

    public float Confidence => Score.Length > 0 ? Score.Max() : 0;
}
