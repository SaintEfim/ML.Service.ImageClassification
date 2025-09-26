namespace ML.Service.ImageClassification.Minio.Models;

public class MinioModel
{
    public required string Name { get; set; }

    public required Stream File { get; set; }

    public required string StoredName { get; set; }

    public required string MinioType { get; set; }
}
