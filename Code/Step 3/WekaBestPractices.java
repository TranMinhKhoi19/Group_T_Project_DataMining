// WekaBestPractices.java
// Best-practice pipeline: filtered classifiers (resample & standardize inside CV),
// repeated stratified CV, multiple metrics, cost-sensitive option, save final models.
//
// Compile:
// javac -cp ".;path\to\weka.jar" WekaBestPractices.java
// Run:
// java --add-opens java.base/java.lang=ALL-UNNAMED -Xmx4g -cp ".;path\to\weka.jar" WekaBestPractices

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.supervised.instance.Resample;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;

import weka.core.SerializationHelper;
import weka.core.Utils;

import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class WekaBestPractices {

    // load ARFF
    public static Instances loadData(String path) throws Exception {
        DataSource src = new DataSource(path);
        Instances data = src.getDataSet();
        if (data == null)
            throw new Exception("No data loaded: " + path);
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // ensure class nominal
    public static Instances ensureClassNominal(Instances data) throws Exception {
        if (data.classAttribute().isNumeric()) {
            weka.filters.unsupervised.attribute.NumericToNominal conv = new weka.filters.unsupervised.attribute.NumericToNominal();
            conv.setAttributeIndices(Integer.toString(data.classIndex() + 1));
            conv.setInputFormat(data);
            Instances out = Filter.useFilter(data, conv);
            out.setClassIndex(out.numAttributes() - 1);
            return out;
        }
        return data;
    }

    // helper: build a FilteredClassifier that applies Resample then Standardize
    // then base classifier
    public static FilteredClassifier buildFilteredClassifier(Classifier base,
            double resamplePercent,
            boolean noReplacement,
            boolean doStandardize) throws Exception {
        // If no resampling wanted, you can set resamplePercent = 100.0 and
        // noReplacement = true to be neutral.
        Resample res = new Resample();
        res.setNoReplacement(noReplacement);
        res.setBiasToUniformClass(1.0);
        res.setSampleSizePercent(resamplePercent);
        // Standardize filter
        Standardize std = new Standardize();

        // We want: outer filter = resample, inner filter = standardize (if set)
        // FilteredClassifier supports only a single filter. To chain, we nest
        // FilteredClassifier:
        FilteredClassifier inner = new FilteredClassifier();
        if (doStandardize) {
            inner.setFilter(std);
        }
        inner.setClassifier(base);

        FilteredClassifier outer = new FilteredClassifier();
        outer.setFilter(res);
        outer.setClassifier(inner);

        return outer;
    }

    // run repeated stratified CV and aggregate metrics
    public static void repeatedCV(Instances data, Classifier cls, int folds, int repeats, int seed) throws Exception {
        double sumAcc = 0;
        double sumAUCpos = 0;
        double sumRecallPos = 0;
        double sumPrecPos = 0;
        double sumF1Pos = 0;
        int runs = 0;

        for (int r = 0; r < repeats; r++) {
            int thisSeed = seed + r;
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, folds, new Random(thisSeed));
            double acc = eval.pctCorrect();
            double aucPos = Double.NaN;
            try {
                aucPos = eval.areaUnderROC(1);
            } catch (Exception e) {
            }
            double recallPos = eval.recall(1);
            double precPos = eval.precision(1);
            double f1Pos = eval.fMeasure(1);

            System.out.printf(
                    "Repeat %d (seed=%d) -> Acc=%.2f%%  Recall_pos=%.3f  Prec_pos=%.3f  F1_pos=%.3f  AUC_pos=%.3f%n",
                    r + 1, thisSeed, acc, recallPos, precPos, f1Pos, Double.isNaN(aucPos) ? Double.NaN : aucPos);

            sumAcc += acc;
            if (!Double.isNaN(aucPos))
                sumAUCpos += aucPos;
            sumRecallPos += recallPos;
            sumPrecPos += precPos;
            sumF1Pos += f1Pos;
            runs++;
        }

        System.out.println("----- Summary across repeats -----");
        System.out.printf("Avg Accuracy = %.2f%%\n", sumAcc / runs);
        System.out.printf("Avg Recall (pos) = %.4f\n", sumRecallPos / runs);
        System.out.printf("Avg Precision (pos) = %.4f\n", sumPrecPos / runs);
        System.out.printf("Avg F1 (pos) = %.4f\n", sumF1Pos / runs);
        if (sumAUCpos > 0)
            System.out.printf("Avg AUC (pos) = %.4f\n", sumAUCpos / runs);
    }

    // Evaluate once and print confusion matrix + class details (single run)
    public static void evalAndPrintSingle(Instances data, Classifier cls, int folds, int seed) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, folds, new Random(seed));
        System.out.println(eval.toSummaryString("\nSummary\n========\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }

    // Train classifier on full data and save model
    public static void trainAndSave(Classifier cls, Instances data, String modelName) throws Exception {
        cls.buildClassifier(data);
        SerializationHelper.write(modelName, cls);
        System.out.println("Saved model -> " + modelName);
    }

    public static void main(String[] args) {
        try {
            String arff = "heart_clean.arff";
            int folds = 5;
            int repeats = 5;
            int seed = 1;

            System.out.println("Loading: " + arff);
            Instances data = loadData(arff);
            data = ensureClassNominal(data);

            // print distribution
            int clsIdx = data.classIndex();
            int[] counts = data.attributeStats(clsIdx).nominalCounts;
            System.out.println("Class distribution:");
            for (int i = 0; i < counts.length; i++) {
                System.out.printf("  %s : %d%n", data.classAttribute().value(i), counts[i]);
            }

            // ----------------------------
            // 1) RandomForest (baseline) - original data
            // ----------------------------
            RandomForest rf = new RandomForest();
            rf.setNumIterations(300);
            System.out.println("\n==== RandomForest (baseline) - original data ====");
            repeatedCV(data, rf, folds, repeats, seed);
            // train final
            trainAndSave(rf, data, "RF_baseline.model");

            // ----------------------------
            // 2) CostSensitive RandomForest - penalize misclassifying positive
            // ----------------------------
            CostSensitiveClassifier csc = new CostSensitiveClassifier();
            RandomForest rf2 = new RandomForest();
            rf2.setNumIterations(300);
            CostMatrix cm = new CostMatrix(2);
            cm.setElement(0, 1, 1.0);
            cm.setElement(1, 0, 5.0); // tune if needed
            csc.setClassifier(rf2);
            csc.setCostMatrix(cm);
            csc.setMinimizeExpectedCost(true);
            System.out.println("\n==== CostSensitive RandomForest (original data) ====");
            repeatedCV(data, csc, folds, repeats, seed);
            trainAndSave(csc, data, "RF_costsensitive.model");

            // ----------------------------
            // 3) Filtered SMO (Resample inside CV + Standardize)
            // ----------------------------
            SMO smo = new SMO();
            // linear kernel (faster and stable); for RBF, you'd tune gamma/C
            weka.classifiers.functions.supportVector.PolyKernel linear = new weka.classifiers.functions.supportVector.PolyKernel();
            linear.setExponent(1.0);
            smo.setKernel(linear);
            smo.setC(1.0);

            // Build filtered classifier (resamplePercent 200 -> oversample minority inside
            // CV)
            FilteredClassifier fcSMO = buildFilteredClassifier(smo, 200.0, false, true);
            System.out.println("\n==== Filtered SMO (resample inside CV + standardize) ====");
            repeatedCV(data, fcSMO, folds, repeats, seed);
            trainAndSave(fcSMO, data, "SMO_filtered.model");

            // ----------------------------
            // 4) Filtered AdaBoost (Resample inside CV)
            // ----------------------------
            AdaBoostM1 ada = new AdaBoostM1();
            ada.setClassifier(new J48());
            ada.setNumIterations(50);
            FilteredClassifier fcAda = buildFilteredClassifier(ada, 200.0, false, false); // no std needed
            System.out.println("\n==== Filtered AdaBoost (resample inside CV) ====");
            repeatedCV(data, fcAda, folds, repeats, seed);
            trainAndSave(fcAda, data, "AdaBoost_filtered.model");

            // ----------------------------
            // 5) NOTES about hyperparameter tuning (optional)
            // ----------------------------
            System.out.println(
                    "\nNOTE: For full rigor, perform nested CV for hyperparameter tuning (CVParameterSelection) " +
                            "or use Weka Experimenter. This code focuses on robust evaluation (resample-in-CV & repeated CV).");

            System.out.println("\nALL DONE.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
