package src;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;

import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("all")
public class datuBilketa {

    private static datuBilketa nireDB = null;

    public static datuBilketa getDB() {
        if (nireDB == null) {
            nireDB = new datuBilketa();
        }
        return nireDB;
    }

    public Instances[] bildu(String inPath, String outFile) {
        // Sarrerako datuak irakurri
        Instances train = datuakBildu(inPath + "/train");
        save(train, outFile + "Train.arff"); // Gorde datuak
        // Aldatu BoW formatura
        Instances trainBoW = NonSparseBoW.getNonSparseBoW().transformTrain(train);
        save(trainBoW, outFile + "TrainBoW.arff"); // Gorde datuak

        // Dev datuak irakurri eta randomizatu (klaseekin)
        Instances devWithClass = datuakBildu(inPath + "/dev");
        save(devWithClass, outFile + "DevWithClass.arff"); // Gorde datuak
        Instances devWithClassBoW = NonSparseBoW.getNonSparseBoW().transformDevTest(devWithClass);
        save(devWithClassBoW, outFile + "DevWithClassBoW.arff"); // Gorde datuak

        // Dev sortu (klaseak ezezagunak jarriz)
        Instances dev = createDev(devWithClass);
        save(dev, outFile + "Dev.arff"); // Gorde datuak
        Instances devBow = createDev(devWithClassBoW);
        save(devBow, outFile + "DevBoW.arff"); // Gorde datuak

        // Test datuak irakurri
        Instances test = datuakBilduTest(inPath + "/test_blind");
        save(test, outFile + "Test.arff"); // Gorde datuak
        Instances testBoW = NonSparseBoW.getNonSparseBoW().transformDevTest(test);
        save(testBoW, outFile + "TestBoW.arff"); //Gorde datuak
        
        return new Instances[] {trainBoW, devBow, devWithClassBoW, testBoW};        
    }

    private Instances datuakBildu(String inPath) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Karpeten izenak klase bezala
            }
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances data = new Instances("Dataset", attributes, 0);
            data.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri
                        StringBuilder contentBuilder = new StringBuilder();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                contentBuilder.append(line).append("\n");
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                        instance.setValue(attributes.get(1), folder.getName()); // Klasea (karpetaren izena)
                        data.add(instance);
                    }
                }
            }
            return data;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    private Instances datuakBilduDev(String inPath) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Karpeten izenak klase bezala
            }
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances data = new Instances("Dataset", attributes, 0);
            data.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri
                        StringBuilder contentBuilder = new StringBuilder();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                contentBuilder.append(line).append("\n");
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                        data.add(instance);
                    }
                }
            }
            return data;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    private Instances datuakBilduTest(String inPath) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            classValues.add("neg");
            classValues.add("pos");
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances dataTest = new Instances("Dataset", attributes, 0);
            dataTest.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            File[] files = dir.listFiles(File::isFile);
            if (files != null) {
                for (File file : files) {
                    // Fitxategiaren edukia irakurri
                    StringBuilder contentBuilder = new StringBuilder();
                    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            contentBuilder.append(line).append("\n");
                        }
                    } catch (Exception e) {
                        System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                        e.printStackTrace();
                    }
                    // Instantzia sortu eta datuak gehitu
                    DenseInstance instance = new DenseInstance(2);
                    instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                    dataTest.add(instance);
                }
            }
            return dataTest;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    private void save(Instances attributes, String outFile) {
        try {
            // datuak karpeta eratu
            File dir = new File("datuak");
            if (!dir.exists()) {
                dir.mkdir();
            }
            FileWriter writer = new FileWriter("datuak/" + outFile);
            // .arff fitxategia gorde
            writer.write(attributes.toString());
            writer.close();
            System.out.println("Fitxategia .arff formatuan ongi sortu da hemen: " + outFile);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void save(String[] data, String outFile) {
        try {
            FileWriter writer = new FileWriter("datuak/" + outFile);
            for (int i = 0; i < data.length; i++) {
                writer.write(data[i] + "\n");
            }
            writer.close();
            System.out.println("Fitxategia .txt formatuan ongi sortu da hemen: " + outFile);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            System.exit(1);
            e.printStackTrace();
        }
    }

    private Instances createDev(Instances dev) {
        try {
            // Create a copy of the Dev dataset
            Instances dev2 = new Instances(dev);

            // Replace all class values with unknown values (?)
            int classIndex = dev2.classIndex();
            for (int i = 0; i < dev2.numInstances(); i++) {
                dev2.instance(i).setMissing(classIndex);
            }

            return dev2;
        } catch (Exception e) {
            System.out.println("Errorea: Ezin izan da Dev2 sortu.");
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public static void main(String[] args) {
        // Test
        String inPath = "movies_reviews/";
        String outFile = "datuak";
        datuBilketa.getDB().bildu(inPath, outFile);
    }
}