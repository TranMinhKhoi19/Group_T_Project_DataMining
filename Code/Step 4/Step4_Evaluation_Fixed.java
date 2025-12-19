
// Step4_Evaluation_Fixed.java
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CostSensitiveClassifier;

import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.util.Random;

public class Step4_Evaluation_Fixed {

    // Load dataset
    public static Instances load(String path) throws Exception {
        DataSource src = new DataSource(path);
        Instances data = src.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // Force class attribute to nominal if numeric
    public static Instances ensureClassNominal(Instances data) throws Exception {
        if (data.classAttribute().isNumeric()) {
            System.out.println("⚠ Class is numeric → Converting to nominal...");
            NumericToNominal conv = new NumericToNominal();
            String idx = Integer.toString(data.classIndex() + 1); // 1-based index
            conv.setAttributeIndices(idx);
            conv.setInputFormat(data);
            Instances newData = Filter.useFilter(data, conv);
            newData.setClassIndex(newData.numAttributes() - 1);
            System.out.println("✔ Class successfully converted to nominal.");
            return newData;
        }
        System.out.println("✔ Class already nominal.");
        return data;
    }

    // Evaluate & print metrics with runtime
    public static void evaluateModel(String name, Classifier cls, Instances data) throws Exception {
        System.out.println("\n==============================");
        System.out.println("Evaluating: " + name);
        System.out.println("==============================");

        long start = System.currentTimeMillis();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));
        long end = System.currentTimeMillis();
        long runtime = end - start;

        System.out.println(eval.toSummaryString("\nSummary\n--------\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        // Print AUC
        System.out.println("AUC values:");
        for (int i = 0; i < data.numClasses(); i++) {
            try {
                System.out.printf("  Class %s: AUC = %.4f%n",
                        data.classAttribute().value(i),
                        eval.areaUnderROC(i));
            } catch (Exception e) {
                System.out.println("  AUC not available for this classifier.");
            }
        }

        System.out.println("Runtime: " + runtime + " ms");

        // Save trained model on full dataset
        cls.buildClassifier(data);
        SerializationHelper.write(name + ".model", cls);
        System.out.println("Saved model -> " + name + ".model");
    }

    public static void main(String[] args) {
        try {
            String path = "heart_clean.arff";
            System.out.println("Loading dataset: " + path);

            Instances data = load(path);
            data = ensureClassNominal(data);

            System.out.println("\nDataset Details:");
            System.out.println("Instances = " + data.numInstances());
            System.out.println("Attributes = " + data.numAttributes());
            System.out.println("Class attribute = " + data.classAttribute().name());
            for (int i = 0; i < data.classAttribute().numValues(); i++)
                System.out.println("  class " + i + ": " + data.classAttribute().value(i));

            // Baseline models
            evaluateModel("J48", new J48(), data);
            evaluateModel("NaiveBayes", new NaiveBayes(), data);

            RandomForest rf = new RandomForest();
            rf.setNumIterations(200);
            evaluateModel("RandomForest", rf, data);

            evaluateModel("Logistic", new Logistic(), data);

            // Improvement models
            evaluateModel("SMO", new SMO(), data);

            AdaBoostM1 boost = new AdaBoostM1();
            boost.setNumIterations(50);
            evaluateModel("AdaBoost", boost, data);

            CostSensitiveClassifier costRF = new CostSensitiveClassifier();
            costRF.setClassifier(new RandomForest());
            evaluateModel("CostSensitiveRF", costRF, data);

            System.out.println("\nALL MODELS COMPLETED.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
