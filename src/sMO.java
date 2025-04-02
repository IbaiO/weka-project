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
        // Cargar los datos de entrenamiento
        Instances train = dataset;
        if (train == null) {
            System.out.println("Error: Unable to load training data.");
            return null;
        }

        // Preprocesar los datos si contienen atributos de texto
        dataset = preprocessData(train);

        // Configurar índice de clase
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }

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
        double[] cValues = {0.1, 1, 10, 100}; // Valores de C a probar
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
            // StringToWordVector iragazkia aplikatu, behar izanez gero.
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

            // Atributu bitar nominalak zenbakizkotan bihurtu
            if (dataset.classIndex() != -1) {
                int classIndex = dataset.classIndex();
                if (dataset.attribute(classIndex).isNominal() && dataset.attribute(classIndex).numValues() == 2) {
                    // Atributu zerrenda bat sortu, DataSet berriarentzat
                    ArrayList<Attribute> attributes = new ArrayList<>();
                    for (int i = 0; i < dataset.numAttributes(); i++) {
                        if (i == classIndex) {
                            // Reemplazar el atributo de clase nominal con un atributo numérico
                            attributes.add(new Attribute("class"));
                        } else {
                            attributes.add(dataset.attribute(i));
                        }
                    }

                    // DataSet berria sortu Pos/Neg klasea zenbaki gisa jarrita
                    // Pos: 1, Neg: 0
                    Instances newDataset = new Instances(dataset.relationName(), attributes, dataset.numInstances());
                    newDataset.setClassIndex(classIndex);

                    // Instantziak DataSet berrira gehitu
                    for (int i = 0; i < dataset.numInstances(); i++) {
                        DenseInstance newInstance = new DenseInstance(newDataset.numAttributes());
                        newInstance.setDataset(newDataset);

                        for (int j = 0; j < dataset.numAttributes(); j++) {
                            if (j == classIndex) {
                                String classValue = dataset.instance(i).stringValue(classIndex);
                                if (classValue.equalsIgnoreCase("pos")) {
                                    newInstance.setValue(classIndex, 1);
                                } else if (classValue.equalsIgnoreCase("neg")) {
                                    newInstance.setValue(classIndex, 0);
                                }
                            } else {
                                newInstance.setValue(j, dataset.instance(i).value(j));
                            }
                        }

                        newDataset.add(newInstance);
                    }

                    dataset = newDataset;
                }
            }

            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu.");
            e.printStackTrace();
            return null;
        }
    }
}

