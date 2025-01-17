﻿using System.Xml.Serialization;

namespace NeuralNetwork;

public class NextGen
{
    [XmlIgnore] private double[]? _curDeltas;
    [XmlIgnore] private double[]? _lastDeltas;

    public NextGen(Func<double, double> activationFunc, Func<double, double> derivativeFunction,
        params NeuronLayer[] layers)
    {
        SetFuncs(activationFunc, derivativeFunction);

        CreateNetwork(layers);
    }

    protected NextGen()
    {
    }

    public required string Name { get; set; }
    private Func<double, double>? ActivationFunction { get; set; }
    private Func<double, double>? DerivativeFunction { get; set; }
    public required Layer[] Layers { get; set; }
    [XmlIgnore] public double LearningRatio { get; set; }

    public void ForwardPassData(double[] input, ref double[] output)
    {
        if(ActivationFunction == null)
            throw new NullReferenceException("You must call SetFuncs first");
        Array.Copy(input, Layers[0].ActivatedNeurons, input.Length);
        for (var layerIndex = 0; layerIndex < Layers.Length - 1; layerIndex++)
        {
            var curL = Layers[layerIndex];
            var nextL = Layers[layerIndex + 1];
            for (var nextN = 0; nextN < nextL.Neurons.Length; nextN++)
            {
                nextL.Neurons[nextN] = 0;
                for (var curN = 0; curN < curL.ActivatedNeurons.Length; curN++)
                    nextL.Neurons[nextN] += curL.ActivatedNeurons[curN] * curL.Weights[curN][nextN];
            }

            nextL.Activate(ActivationFunction);
        }

        Array.Copy(Layers[^1].ActivatedNeurons, output, output.Length);
    }

    public double AdjustWeights(double[] input, double[] targets, ref double[] output)
    {
        if (_curDeltas == null || _lastDeltas == null)
            throw new NullReferenceException("You must call InitLearn first");
        if(DerivativeFunction == null)
            throw new NullReferenceException("You must call SetFuncs first");
        ForwardPassData(input, ref output);

        for (var i = 0; i < Layers[^1].NumOfInputNeurons; i++)
            _lastDeltas[i] = (targets[i] - Layers[^1].ActivatedNeurons[i]) *
                             DerivativeFunction(Layers[^1].Neurons[i]);
        for (var layer = Layers.Length - 2; layer >= 0; layer--)
        {
            for (var n = 0; n < Layers[layer].Neurons.Length; n++)
            {
                var delta = 0.0;
                var gradient = LearningRatio * Layers[layer].ActivatedNeurons[n];
                for (var nextN = 0; nextN < Layers[layer + 1].NumOfInputNeurons; nextN++)
                {
                    delta += _lastDeltas[nextN] * Layers[layer].Weights[n][nextN];
                    Layers[layer].Weights[n][nextN] += gradient * _lastDeltas[nextN];
                }
                _curDeltas[n] = delta * DerivativeFunction(Layers[layer].Neurons[n]);
            }

            (_lastDeltas, _curDeltas) = (_curDeltas, _lastDeltas);
        }

        double sum = 0;
        for (var i = 0; i < targets.Length; i++)
            sum += (targets[i] - Layers[^1].ActivatedNeurons[i]) *
                   (targets[i] - Layers[^1].ActivatedNeurons[i]);
        return sum / targets.Length;
    }

    public void InitLearn()
    {
        var maxNeurons = 0;
        foreach (var layer in Layers)
            maxNeurons = Math.Max(layer.Neurons.Length, maxNeurons);

        _curDeltas = new double[maxNeurons];
        _lastDeltas = new double[maxNeurons];
    }

    public void SetFuncs(Func<double, double> activationFunc, Func<double, double> derivativeFunction)
    {
        ActivationFunction = activationFunc;
        DerivativeFunction = derivativeFunction;
    }

    private void CreateNetwork(NeuronLayer[] layers)
    {
        Layers = new Layer[layers.Length];
        int curLayer;
        for (curLayer = 0; curLayer < layers.Length - 1; curLayer++)
            Layers[curLayer] = new Layer(layers[curLayer].NumOfNeurons, layers[curLayer + 1].NumOfNeurons,
                layers[curLayer].Bias);
        Layers[curLayer] = new Layer(layers[curLayer].NumOfNeurons, 0, layers[curLayer].Bias);
    }

    public static void SaveToFile(NextGen net, string path)
    {
        var xmlSerializer = new XmlSerializer(typeof(NextGen));
        xmlSerializer.Serialize(File.Create(path), net);
    }

    public static NextGen LoadFromFile(string path)
    {
        var xmlSerializer = new XmlSerializer(typeof(NextGen));
        return (NextGen)xmlSerializer.Deserialize(File.OpenRead(path))!;
    }

    public override string ToString()
    {
        var res = $"{Name} [";
        for (var i = 0; i < Layers.Length; i++)
        {
            res += Layers[i].NumOfInputNeurons.ToString();
            if (i < Layers.Length - 1)
                res += " ";
        }

        res += "]";
        return res;
    }
}

public class Layer
{
    public Layer(int size, int nextSize, bool bias)
    {
        Bias = bias;

        Neurons = new double[size];
        ActivatedNeurons = new double[Neurons.Length + (Bias ? 1 : 0)];
        if (Bias)
            ActivatedNeurons[Neurons.Length - 1] = 1;

        Weights = new double[ActivatedNeurons.Length][];
        for (var i = 0; i < Weights.Length; i++)
            Weights[i] = RandomArray(nextSize);
    }
    
    protected Layer() {}

    public int NumOfInputNeurons => Neurons.Length - (Bias ? 1 : 0);
    public double[] Neurons { get; set; }
    public double[] ActivatedNeurons { get; set; }
    public double[][] Weights { get; set; }
    public bool Bias { get; set; }

    public void Activate(Func<double, double> act)
    {
        for (var i = 0; i < Neurons.Length; i++)
            ActivatedNeurons[i] = act(Neurons[i]);
    }

    private double[] RandomArray(int length)
    {
        var answer = new double[length];
        var rnd = new Random();
        for (var i = 0; i < length; i++)
            answer[i] = rnd.NextDouble() * 2 - 1;
        return answer;
    }
}

public struct NeuronLayer
{
    public int NumOfNeurons { get; set; }
    public bool Bias { get; set; }

    public NeuronLayer(int neurons, bool bias = false)
    {
        NumOfNeurons = neurons;
        Bias = bias;
    }
}