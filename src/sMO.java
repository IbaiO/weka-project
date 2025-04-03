package src;

import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.ArrayList;
import java.util.Random;

public class sMO {

    public static SMO[] main(Instances dataset) throws Exception {
    
        // Ignorar advertencias de netlib-java y establecer propiedades para usar F2jBLAS y F2jLAPACK
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        // Cargar los datos de entrenamiento
        Instances train = dataset;
        if (train == null) {
            System.out.println("Error: Unable to load training data.");
            return null;
        }

        System.out.println("Índice de clase: " + train.classIndex());
        System.out.println("Nombre del atributo de clase: " + train.attribute(train.classIndex()).name());
        System.out.println("Tipo del atributo de clase: " + (train.attribute(train.classIndex()).isNominal() ? "Nominal" : "Numérico"));

        // Preprocesar los datos si contienen atributos de texto
        train = preprocessData(train);

        // Comparar diferentes kernels y optimizar C
        System.out.println("Evaluating PolyKernel...");
        SMO polyKernelModel = evaluateKernel(train, new weka.classifiers.functions.supportVector.PolyKernel(), 2); // E = 2

        System.out.println("Evaluating RBFKernel...");
        SMO rbfKernelModel = evaluateKernel(train, new weka.classifiers.functions.supportVector.RBFKernel(), 0.01); // Gamma = 0.1

        System.out.println("Evaluating PukKernel...");
        SMO pukKernelModel = evaluateKernel(train, new weka.classifiers.functions.supportVector.Puk(), 1.0); // Omega = 1.0


        // Crear un arreglo para almacenar los modelos evaluados
        SMO[] models = {polyKernelModel, rbfKernelModel, pukKernelModel};

        // Devolver ambos modelos
        return models;
    }

    private static SMO evaluateKernel(Instances train, weka.classifiers.functions.supportVector.Kernel kernel, double kernelParam) throws Exception {
        double[] cValues = {0.1, 1, 10}; // Valores de C a probar
        double[] gammaValues = {0.01, 0.1, 1}; // Valores de Gamma (para RBFKernel)
        double[] exponentValues = {1, 2, 3}; // Valores de Exponent (para PolyKernel)
        double[] omegaValues = {0.5, 1.0, 2.0}; // Valores de Omega (para PukKernel)
        double[] sigmaValues = {0.5, 1.0, 2.0}; // Valores de Sigma (para PukKernel)

        double bestWeightedFMeasure = 0; // Mejor weighted average F-measure
        double bestC = 0;
        double bestKernelParam1 = 0; // Puede ser Gamma, Exponent, Omega, etc.
        double bestKernelParam2 = 0; // Solo para PukKernel (Sigma)
        SMO bestModel = null;

        for (double c : cValues) {
            for (double param1 : (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel ? gammaValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.PolyKernel ? exponentValues :
                                  kernel instanceof weka.classifiers.functions.supportVector.Puk ? omegaValues : new double[]{kernelParam})) {
                for (double param2 : (kernel instanceof weka.classifiers.functions.supportVector.Puk ? sigmaValues : new double[]{1.0})) {
                    // Crear el clasificador SMO
                    SMO smo = new SMO();
                    smo.setKernel(kernel);
                    smo.setC(c);

                    // Configurar parámetros específicos del kernel
                    if (kernel instanceof weka.classifiers.functions.supportVector.PolyKernel) {
                        ((weka.classifiers.functions.supportVector.PolyKernel) kernel).setExponent((int) param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.RBFKernel) {
                        ((weka.classifiers.functions.supportVector.RBFKernel) kernel).setGamma(param1);
                    } else if (kernel instanceof weka.classifiers.functions.supportVector.Puk) {
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setOmega(param1);
                        ((weka.classifiers.functions.supportVector.Puk) kernel).setSigma(param2);
                    }

                    // Evaluar el modelo con validación cruzada
                    Evaluation eval = new Evaluation(train);
                    eval.crossValidateModel(smo, train, 10, new Random(1)); // 10-fold cross-validation

                    // Obtener el weighted average F-measure
                    double weightedFMeasure = eval.weightedFMeasure();

                    // Imprimir resultados
                    System.out.println("Kernel: " + kernel.getClass().getSimpleName());
                    System.out.println("C: " + c);
                    System.out.println("Param1: " + param1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Param2: " + param2 : ""));
                    System.out.println("Weighted F-Measure: " + weightedFMeasure);
                    System.out.println("====================================");

                    // Actualizar el mejor modelo si el weighted F-measure mejora
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

        // Entrenar el mejor modelo con el conjunto de entrenamiento completo
        if (bestModel != null) {
            bestModel.buildClassifier(train);
        }

        // Imprimir el mejor valor de parámetros
        System.out.println("Best C for " + kernel.getClass().getSimpleName() + ": " + bestC);
        System.out.println("Best Param1: " + bestKernelParam1 + (kernel instanceof weka.classifiers.functions.supportVector.Puk ? ", Best Param2: " + bestKernelParam2 : ""));
        System.out.println("Best Weighted F-Measure: " + bestWeightedFMeasure);
        System.out.println("====================================");

        // Retornar el mejor modelo con los mejores parámetros
        return bestModel;
    }

    private static Instances preprocessData(Instances dataset) {
        try {
            // Configurar el índice de clase como el primer atributo
            dataset.setClassIndex(0);

            // Verificar si el atributo de clase es numérico y convertirlo a nominal si es necesario
            if (dataset.classIndex() != -1 && dataset.attribute(dataset.classIndex()).isNumeric()) {
                dataset = convertNumericClassToNominal(dataset, dataset.classIndex());
            }

            // Aplicar StringToWordVector si hay atributos de texto
            boolean hasStringAttributes = false;
            for (int i = 0; i < dataset.numAttributes(); i++) {
                if (dataset.attribute(i).isString()) {
                    hasStringAttributes = true;
                    break;
                }
            }

            if (hasStringAttributes) {
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(dataset);
                dataset = Filter.useFilter(dataset, stringToWordVector);
            }

            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu.");
            e.printStackTrace();
            return null;
        }
    }
    
    private static Instances convertNumericClassToNominal(Instances dataset, int classIndex) throws Exception {
        weka.filters.unsupervised.attribute.NumericToNominal numericToNominal = new weka.filters.unsupervised.attribute.NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(classIndex + 1)); // Weka usa índices 1-based
        numericToNominal.setInputFormat(dataset);
        return Filter.useFilter(dataset, numericToNominal);
    }
}

