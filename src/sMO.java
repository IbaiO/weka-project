package src;

import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.Random;

public class sMO {

    public static SMO[] main(Instances dataset) throws Exception {
        // Netlib-java-ren abisuak ezabatu eta F2jBLAS eta F2jLAPACK erabiltzeko propietateak ezarri
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        if (dataset == null) {
            System.out.println("Errorea: Ezin izan da entrenamendu datuak kargatu."); // Error: Unable to load training data
            return null;
        }

        // Datuak orekatzea klaseak uniformeki banatzeko
        weka.filters.supervised.instance.Resample resample = new weka.filters.supervised.instance.Resample();
        resample.setBiasToUniformClass(1.0); // Klaseak orekatzea emaitza hobeak lortzeko
        resample.setInputFormat(dataset);
        Instances balancedTrainSet = Filter.useFilter(dataset, resample);

        balancedTrainSet.setClassIndex(0); // Klase atributuaren indizea ezarri

        // Kernel desberdinak konparatu eta C optimizatu
        System.out.println("PolyKernel ebaluatzen...");
        SMO polyKernelModel = evaluateKernel(balancedTrainSet, new weka.classifiers.functions.supportVector.PolyKernel(), 2); // E = 2

        System.out.println("RBFKernel ebaluatzen...");
        SMO rbfKernelModel = evaluateKernel(balancedTrainSet, new weka.classifiers.functions.supportVector.RBFKernel(), 0.01); // Gamma = 0.1

        System.out.println("PukKernel ebaluatzen...");
        SMO pukKernelModel = evaluateKernel(balancedTrainSet, new weka.classifiers.functions.supportVector.Puk(), 1.0); // Omega = 1.0

        // Ebaluatutako modeloak gordetzeko array bat sortu
        SMO[] models = {polyKernelModel, rbfKernelModel, pukKernelModel};

        // Modeloak itzuli
        return models;
    }

    private static SMO evaluateKernel(Instances train, weka.classifiers.functions.supportVector.Kernel kernel, double kernelParam) throws Exception {
        double[] cValues = {0.1, 1, 10}; // C-ren balioen tartea zabaldu
        double[] gammaValues = {0.0001, 0.001, 0.01, 0.1}; // Gamma-ren balioen tartea zabaldu
        double[] exponentValues = {1, 2, 3}; // Exponent balioak (PolyKernel-erako)
        double[] omegaValues = {0.5, 1.0, 2.0}; // Omega balioak (PukKernel-erako)
        double[] sigmaValues = {0.5, 1.0, 2.0}; // Sigma balioak (PukKernel-erako)

        double bestWeightedFMeasure = 0; // Weighted average F-measure onena
        double bestC = 0;
        double bestKernelParam1 = 0; // Gamma, Exponent, Omega, etab.
        double bestKernelParam2 = 0; // Sigma (PukKernel-erako soilik)
        SMO bestModel = null;

        for (double c : cValues) {
            for (double param1 : (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel ? gammaValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.PolyKernel ? exponentValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.Puk ? omegaValues : new double[]{kernelParam})) {
                for (double param2 : (kernel instanceof weka.classifiers.functions.supportVector.Puk ? sigmaValues : new double[]{1.0})) {
                    // SMO klasifikatzailea sortu
                    SMO smo = new SMO();
                    smo.setKernel(kernel);
                    smo.setC(c);

                    // Kernelaren parametro espezifikoak konfiguratu
                    if (kernel instanceof weka.classifiers.functions.supportVector.PolyKernel) {
                        ((weka.classifiers.functions.supportVector.PolyKernel) kernel).setExponent((int) param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel) {
                        ((weka.classifiers.functions.supportVector.RBFKernel) kernel).setGamma(param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.Puk) {
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setOmega(param1);
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setSigma(param2);
                    }

                    // Modeloa ebaluatu gurutze-balidazioarekin
                    Evaluation eval = new Evaluation(train);
                    eval.crossValidateModel(smo, train, 10, new Random(1)); // 10-fold cross-validation

                    // Weighted average F-measure lortu
                    double weightedFMeasure = eval.weightedFMeasure();

                    // Emaitzak inprimatu
                    System.out.println("Kernel: " + kernel.getClass().getSimpleName());
                    System.out.println("C: " + c);
                    System.out.println("Param1: " + param1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Param2: " + param2 : ""));
                    System.out.println("Weighted F-Measure: " + weightedFMeasure);
                    System.out.println("====================================");

                    // Weighted F-measure hobetzen bada, modeloa eguneratu
                    if (weightedFMeasure > bestWeightedFMeasure) {
                        bestWeightedFMeasure = weightedFMeasure;
                        bestC = c;
                        bestKernelParam1 = param1;
                        bestKernelParam2 = param2;
                        bestModel = smo;
                    }
                }
            }
        }

        // Modelo onena entrenamendu multzo osoarekin entrenatu
        if (bestModel != null) {
            bestModel.buildClassifier(train);
        }

        // Parametro onenen balioak inprimatu
        System.out.println("Best C for " + kernel.getClass().getSimpleName() + ": " + bestC);
        System.out.println("Best Param1: " + bestKernelParam1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Best Param2: " + bestKernelParam2 : ""));
        System.out.println("Best Weighted F-Measure: " + bestWeightedFMeasure);
        System.out.println("====================================");

        // Parametro onenekin modeloa itzuli
        return bestModel;
    }
}