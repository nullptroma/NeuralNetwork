using NeuralNetwork;

var neuro = NextGen.LoadFromFile("/home/nullptr/neuro.xml");
if (neuro == null)
{
    Console.WriteLine("Error");
    return 1;
}
neuro.SetFuncs(Activation, Derivation);
var data = new double[] { 0.2, 0.8 };
var expected = new double[] { 0.6, 0.4 };
neuro.LearningRatio = 0.3;
neuro.InitLearn();
var output = new double[2];
neuro.AdjustWeights(data, expected, ref output);
neuro.AdjustWeights(data, expected, ref output);
Console.WriteLine(string.Join(", ", output));
NextGen.SaveToFile(neuro, "/home/nullptr/neuro3.xml");
return 0;

double Activation(double x) => (Math.Exp(6*x)-1)/(Math.Exp(6*x)+1);
double Derivation(double x) => 1/Math.Pow(Math.Cosh(3*x), 2);

