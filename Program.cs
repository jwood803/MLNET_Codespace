using Microsoft.ML;
using Microsoft.ML.Data;

var context = new MLContext(seed: 42);

var trainData = context.Data.LoadFromTextFile<SalaryData>("./SalaryData.csv", hasHeader: true, separatorChar: ',');

var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
    .Append(context.Regression.Trainers.LbfgsPoissonRegression());

var model = pipeline.Fit(trainData);

var predictions = model.Transform(trainData);

var metrics = context.Regression.Evaluate(predictions);

Console.WriteLine($"R^2 - {metrics.RSquared}");

class SalaryData
{
        [LoadColumn(0)]
        public float YearsExperience;

        [LoadColumn(1), ColumnName("Label")]
        public float Salary;

}