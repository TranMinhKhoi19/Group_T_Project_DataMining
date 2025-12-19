// Step4_Evaluation.java
// Evaluate multiple models using 10-fold CV + measure runtime

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.meta.FilteredClassifier;

import weka.classifiers.CostMatrix;

import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Standardize;

import weka.core.SerializationHelper;

import java.util.Random;

public class Step4_Evaluation {

    // Load dataset
    public static Instances load(String path) throws Exception {
        Instances data = DataSource.read(path);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // Evaluate with 10-fold CV + measure time
    public static void evaluateModel(Instances data, Classifier model, String name) throws Exception {
        System.out.println("\n==============================");
        System.out.println("Evaluating: " + name);
        System.out.println("==============================");

        long startTrain = System.currentTimeMillis();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));
        long endTrain = System.currentTimeMillis();

        long runtime = endTrain - startTrain;

        System.out.println(eval.toSummaryString("\nSummary:", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        System.out.println("AUC Class 1: " + eval.areaUnderROC(1));
        System.out.println("Run-time (ms): " + runtime);

        // Train final model and save
        model.buildClassifier(data);
        SerializationHelper.write(name + "_final.model", model);
    }

    public static void main(String[] args) {
        try {
            Instances data = load("heart_clean.arff");

            // -------------------------
            // Model 1: J48
            // -------------------------
            evaluateModel(data, new J48(), "J48");

            // -------------------------
            // Model 2: RandomForest
            // -------------------------
            RandomForest rf = new RandomForest();
            rf.setNumIterations(300);
            evaluateModel(data, rf, "RandomForest");

            // -------------------------
            // Model 3: AdaBoost (with J48)
            // -------------------------
            AdaBoostM1 ada = new AdaBoostM1();
            ada.setClassifier(new J48());
            ada.setNumIterations(50);

            // Use FilteredClassifier to avoid leakage
            Resample res = new Resample();
            res.setNoReplacement(false);
            res.setSampleSizePercent(200.0);
            FilteredClassifier fcAda = new FilteredClassifier();
            fcAda.setFilter(res);
            fcAda.setClassifier(ada);

            evaluateModel(data, fcAda, "AdaBoost");

            // -------------------------
            // Model 4: SMO (linear)
            // -------------------------
            SMO smo = new SMO();
            weka.classifiers.functions.supportVector.PolyKernel linear = new weka.classifiers.functions.supportVector.PolyKernel();
            linear.setExponent(1.0);
            smo.setKernel(linear);

            Standardize std = new Standardize();
            FilteredClassifier fcSMO = new FilteredClassifier();
            fcSMO.setFilter(std);
            fcSMO.setClassifier(smo);

            evaluateModel(data, fcSMO, "SMO");

            // -------------------------
            // Model 5: CostSensitive RandomForest
            // -------------------------
            CostSensitiveClassifier csc = new CostSensitiveClassifier();
            RandomForest rf2 = new RandomForest();
            rf2.setNumIterations(300);

            CostMatrix cm = new CostMatrix(2);
            cm.setElement(0, 1, 1.0);
            cm.setElement(1, 0, 5.0);

            csc.setClassifier(rf2);
            csc.setCostMatrix(cm);
            csc.setMinimizeExpectedCost(true);

            evaluateModel(data, csc, "CostSensitiveRF");

            System.out.println("\n=== STEP 4 DONE ===");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
