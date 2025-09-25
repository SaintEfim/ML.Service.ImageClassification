namespace ML.Service.ImageClassification.Minio.Models;

public class MinioModel
{
    public required string Name { get; set; }

    public string? Uri { get; set; }

    public Stream? File;
}
