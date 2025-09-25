using Microsoft.Extensions.Configuration;
using Minio;
using ML.Service.ImageClassification.Minio.Models;

namespace ML.Service.ImageClassification.Model.Minio.Services;

public class MinioProvider : IMinioProvider
{
    private readonly Lazy<IMinioClient> _minioClient;
    private readonly Lazy<string> _bucketName;

    public MinioProvider(
        IConfiguration config)
    {
        _minioClient = new Lazy<IMinioClient>(new MinioClient().WithEndpoint(config["ModelStorage:Minio:Endpoint"])
            .WithCredentials(config["ModelStorage:Minio:AccessKey"], config["ModelStorage:Minio:SecretKey"])
            .Build());

        _bucketName = new Lazy<string>(() =>
            config["ModelStorage:Minio:BucketName"] ?? throw new InvalidOperationException());
    }

    public Task<MinioModel> UploadFile(
        MinioModel model,
        CancellationToken cancellation = default)
    {
        return null;
    }
}
