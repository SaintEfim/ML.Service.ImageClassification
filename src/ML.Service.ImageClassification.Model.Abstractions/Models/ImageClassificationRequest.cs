namespace ML.Service.ImageClassification.Model.Models;

public class ImageClassificationRequest
{
    public byte[] ImageBytes { get; set; } = [];

    public string Label { get; set; } = string.Empty;
}
