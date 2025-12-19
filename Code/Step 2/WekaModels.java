
// WekaModels.java
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;

import weka.core.SerializationHelper;

import java.util.Random;
import java.io.File;

public class WekaModels {

    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // If class attribute is numeric (0/1), convert it to nominal
    public static Instances ensureClassIsNominal(Instances data) throws Exception {
        if (data.classAttribute().isNumeric()) {
            // convert the class attribute (last index) to nominal
            NumericToNominal convert = new NumericToNominal();
            // set attribute indices to last attribute
            String lastIndex = Integer.toString(data.classIndex() + 1); // 1-based
            convert.setAttributeIndices(lastIndex);
            convert.setInputFormat(data);
            Instances newData = Filter.useFilter(data, convert);
            // reset class index
            newData.setClassIndex(newData.numAttributes() - 1);
            return newData;
        } else {
            return data;
        }
    }

    public static void evaluateAndSave(Classifier cls, Instances data, String modelName) throws Exception {
        System.out.println("\n=== Running: " + cls.getClass().getSimpleName() + " ===");

        // 10-fold CV evaluation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));

        System.out.println(eval.toSummaryString("\nSummary\n========\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        // AUC per class (if probabilistic)
        try {
            int classCount = data.numClasses();
            System.out.println("AUC per class:");
            for (int i = 0; i < classCount; i++) {
                double auc = eval.areaUnderROC(i);
                String classValue = data.classAttribute().value(i);
                System.out.printf("  Class %s (index %d): AUC = %.4f%n", classValue, i, auc);
            }
        } catch (Exception e) {
            System.out.println("AUC not available: " + e.getMessage());
        }

        // Train on full data and save model
        cls.buildClassifier(data);
        SerializationHelper.write(modelName, cls);
        System.out.println("Saved model -> " + modelName);
    }

    public static void main(String[] args) {
        try {
            String arffPath = "heart_clean.arff";
            System.out.println("Loading: " + arffPath);
            Instances data = loadData(arffPath);

            // ensure class nominal (important for many classifiers & AUC)
            Instances dataNominal = ensureClassIsNominal(data);

            System.out.println(
                    "Instances: " + dataNominal.numInstances() + "  Attributes: " + dataNominal.numAttributes());
            System.out.println("Class attribute: " + dataNominal.classAttribute().name());
            System.out.println("Class values: ");
            for (int i = 0; i < dataNominal.classAttribute().numValues(); i++) {
                System.out.println("  " + i + " -> " + dataNominal.classAttribute().value(i));
            }

            // 1) J48
            Classifier j48 = new J48(); // bạn có thể set options
            evaluateAndSave(j48, dataNominal, "J48.model");

            // 2) NaiveBayes
            Classifier nb = new NaiveBayes();
            evaluateAndSave(nb, dataNominal, "NaiveBayes.model");

            // 3) RandomForest
            RandomForest rf = new RandomForest();
            rf.setNumIterations(200); // tăng số cây nếu muốn
            evaluateAndSave(rf, dataNominal, "RandomForest.model");

            // 4) Logistic
            Classifier log = new Logistic();
            evaluateAndSave(log, dataNominal, "Logistic.model");

            System.out.println("\nAll models trained and saved.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
