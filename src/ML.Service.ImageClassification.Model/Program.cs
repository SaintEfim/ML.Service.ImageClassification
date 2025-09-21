using Autofac;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using ML.Service.ImageClassification.Model.Services;

namespace ML.Service.ImageClassification.Model;

internal abstract class Program
{
    private static void Main()
    {
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";

        /* var configuration = new ConfigurationBuilder().AddJsonFile("appsettings.json", optional: false)
             .AddJsonFile($"appsettings.{environment}.json", optional: true)
             .Build(); */

        var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            //  .AddConfiguration(configuration.GetSection("Logging"));
        });

        var builder = new ContainerBuilder();

        builder.RegisterType<MLContext>()
            .AsSelf();
        builder.RegisterType<TrainingImageClassification>();
        builder.RegisterInstance(loggerFactory)
            .As<ILoggerFactory>()
            .SingleInstance();
        builder.RegisterGeneric(typeof(Logger<>))
            .As(typeof(ILogger<>))
            .SingleInstance();

        var container = builder.Build();
        container.Resolve<TrainingImageClassification>()
            .TrainModel();
    }
}
