using ML.Service.ImageClassification.Model.Models;

namespace ML.Service.ImageClassification.Model.Services;

public interface IImageClassificationPrediction : IDisposable
{
    ImageClassificationResponse ClassifySingleImage(
        ImageClassificationRequest request);
}
