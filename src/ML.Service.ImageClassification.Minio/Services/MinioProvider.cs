using Microsoft.Extensions.Configuration;
using Minio;
using Minio.DataModel.Args;
using ML.Service.ImageClassification.Minio.Models;
using ML.Service.ImageClassification.Model.Minio.Services;

namespace ML.Service.ImageClassification.Minio.Services;

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

    public async Task UploadFile(
        MinioModel model,
        CancellationToken cancellation = default)
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
}
