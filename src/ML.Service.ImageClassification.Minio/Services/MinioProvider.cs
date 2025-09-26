using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Minio;
using Minio.DataModel.Args;
using Minio.Exceptions;
using ML.Service.ImageClassification.Minio.Models;

namespace ML.Service.ImageClassification.Minio.Services;

public class MinioProvider : IMinioProvider
{
    private readonly Lazy<IMinioClient> _minioClient;
    private readonly Lazy<string> _bucketName;
    private readonly ILogger<MinioProvider> _logger;

    public MinioProvider(
        IConfiguration config,
        ILogger<MinioProvider> logger)
    {
        _logger = logger;
        _minioClient = new Lazy<IMinioClient>(new MinioClient().WithEndpoint(config["ModelStorage:Minio:Endpoint"])
            .WithCredentials(config["ModelStorage:Minio:AccessKey"], config["ModelStorage:Minio:SecretKey"])
            .Build());

        _bucketName = new Lazy<string>(() =>
            config["ModelStorage:Minio:BucketName"] ?? throw new InvalidOperationException());
    }

    public async Task UploadFile(
        MinioModel model,
        CancellationToken cancellation = default)
    {
        try
        {
            var beArgs = new BucketExistsArgs().WithBucket(_bucketName.Value);

            var found = await _minioClient.Value
                .BucketExistsAsync(beArgs, cancellation)
                .ConfigureAwait(false);
            if (!found)
            {
                var mbArgs = new MakeBucketArgs().WithBucket(_bucketName.Value);
                await _minioClient.Value
                    .MakeBucketAsync(mbArgs, cancellation)
                    .ConfigureAwait(false);
            }

            var putObjectArgs = new PutObjectArgs().WithBucket(_bucketName.Value)
                .WithObject(model.StoredName)
                .WithContentType(model.MinioType)
                .WithObjectSize(model.File.Length);

            await _minioClient.Value
                .PutObjectAsync(putObjectArgs, cancellation)
                .ConfigureAwait(false);
        }
        catch (MinioException ex)
        {
            _logger.LogError("File Upload Error: {0}", ex.Message);
        }
    }
}
