using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Vision;
using ML.Service.ImageClassification.Model.Models;

namespace ML.Service.ImageClassification.Model.Services;

internal sealed class TrainingImageClassification
{
    private readonly MLContext _mlContext;
    private readonly ILogger<TrainingImageClassification> _logger;

    private const string DatasetFolder =
        @"..\Dataset";

    private const string ModelFolder =
        @"\TrainedModels";

    public TrainingImageClassification(
        MLContext mlContext,
        ILogger<TrainingImageClassification> logger)
    {
        _mlContext = mlContext;
        _logger = logger;
    }

    public void TrainModel()
    {
        try
        {
            _logger.LogInformation("Starting model training process...");

            var images = LoadImages();
            var dataSplits = SplitData(images);
            LogDataSplitCounts(dataSplits);

            var preprocessingPipeline = CreatePreprocessingPipeline();
            var preprocessor = preprocessingPipeline.Fit(dataSplits.TrainSetRaw);

            var processedData = PreprocessData(preprocessor, dataSplits);
            var trainer = CreateTrainer(processedData.ValidationSet);

            _logger.LogInformation("Training started...");
            var trainerModel = trainer.Fit(processedData.TrainSet);
            _logger.LogInformation("Training finished.");

            var fullModel = preprocessor.Append(trainerModel);
            EvaluateModel(fullModel, dataSplits.TestSet);
            SaveModel(fullModel, dataSplits.TrainSetRaw);

            _logger.LogInformation("Model training completed successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error occurred during model training");
            throw;
        }
    }

    private IDataView LoadImages()
    {
        var files = Directory.GetFiles(DatasetFolder, "*", SearchOption.AllDirectories)
            .Where(f =>
            {
                var ext = Path.GetExtension(f)
                    .ToLowerInvariant();
                return ext is ".jpg" or ".jpeg" or ".png" or ".bmp";
            })
            .Select(path => new ImageData
            {
                ImagePath = path,
                Label = Directory.GetParent(path)!.Name
            })
            .ToList();

        if (files.Count == 0)
        {
            throw new Exception("No images found in dataset folder");
        }

        // Перемешаем для равномерного распределения
        var rnd = new Random(123);
        files = files.OrderBy(_ => rnd.Next())
            .ToList();

        return _mlContext.Data.LoadFromEnumerable(files);
    }

    private DataSplits SplitData(
        IDataView images)
    {
        // 1) Финальный тест 20%
        var outerSplit = _mlContext.Data.TrainTestSplit(images, testFraction: 0.2);
        var trainFull = outerSplit.TrainSet;
        var testSet = outerSplit.TestSet;

        // 2) Из trainFull выделяем validation 10%
        var innerSplit = _mlContext.Data.TrainTestSplit(trainFull, testFraction: 0.1);
        var trainSetRaw = innerSplit.TrainSet;
        var validationSetRaw = innerSplit.TestSet;

        return new DataSplits
        {
            TrainSetRaw = trainSetRaw,
            ValidationSetRaw = validationSetRaw,
            TestSet = testSet
        };
    }

    private void LogDataSplitCounts(
        DataSplits splits)
    {
        _logger.LogInformation("Counts - train: {TrainCount}, val: {ValCount}, test: {TestCount}", _mlContext.Data
            .CreateEnumerable<ImageData>(splits.TrainSetRaw, reuseRowObject: false)
            .Count(), _mlContext.Data
            .CreateEnumerable<ImageData>(splits.ValidationSetRaw, reuseRowObject: false)
            .Count(), _mlContext.Data
            .CreateEnumerable<ImageData>(splits.TestSet, reuseRowObject: false)
            .Count());
    }

    private IEstimator<ITransformer> CreatePreprocessingPipeline()
    {
        return _mlContext.Transforms
            .Conversion
            .MapValueToKey("LabelAsKey", nameof(ImageData.Label))
            .Append(_mlContext.Transforms.LoadRawImageBytes("Image", DatasetFolder, nameof(ImageData.ImagePath)));
    }

    private static ProcessedData PreprocessData(
        ITransformer preprocessor,
        DataSplits splits)
    {
        return new ProcessedData
        {
            TrainSet = preprocessor.Transform(splits.TrainSetRaw),
            ValidationSet = preprocessor.Transform(splits.ValidationSetRaw)
        };
    }

    private IEstimator<ITransformer> CreateTrainer(
        IDataView validationSet)
    {
        var options = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelAsKey",
            ValidationSet = validationSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            Epoch = 50,
            BatchSize = 32,
            LearningRate = 0.01f,
            WorkspacePath = Path.Combine(ModelFolder, "workspace"),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
            MetricsCallback = m => { _logger.LogInformation("Epoch metrics: {Metrics}", m.ToString()); }
        };

        return _mlContext.MulticlassClassification
            .Trainers
            .ImageClassification(options)
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
    }

    private void EvaluateModel(
        ITransformer model,
        IDataView testSet)
    {
        var predictions = model.Transform(testSet);
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelAsKey",
            predictedLabelColumnName: "PredictedLabel");

        _logger.LogInformation("MicroAccuracy: {Micro}", metrics.MicroAccuracy);
        _logger.LogInformation("MacroAccuracy: {Macro}", metrics.MacroAccuracy);
        _logger.LogInformation("LogLoss: {LogLoss}", metrics.LogLoss);
    }

    private void SaveModel(
        ITransformer fullModel,
        IDataView trainSetRaw)
    {
        Directory.CreateDirectory(ModelFolder);
        var modelPath = Path.Combine(ModelFolder, "imageClassifier.zip");
        _mlContext.Model.Save(fullModel, trainSetRaw.Schema, modelPath);
        _logger.LogInformation("Model saved to {ModelPath}", modelPath);
    }
}
