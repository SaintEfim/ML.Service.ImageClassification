using ML.Service.ImageClassification.Minio.Models;

namespace ML.Service.ImageClassification.Minio.Services;

public interface IMinioProvider
{
    Task UploadFile(
        MinioModel model,
        CancellationToken cancellation = default);
}
