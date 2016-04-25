package com.textualinsights.mirchiweb;

import com.google.gson.Gson;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;

import javax.servlet.ServletException;

public class TestSentimentEngine {
    private static final String TEXT_PARAMETER = "text";
    public static final String CLASSIFIER_NAME = "SentimentEngine";
    public static Classifier classifier = null;
    public static Instances trainHeader = null;
    public void init() {
        try {
            if (classifier == null) {
                InputStream fileStream = this.getClass().getClassLoader()
                        .getResourceAsStream(CLASSIFIER_NAME + ".bin");
                if (fileStream != null) {
                    ObjectInputStream objectStream = new ObjectInputStream(fileStream);
                    classifier = (Classifier) objectStream.readObject();
                    objectStream.close();
                }
            }
            if (trainHeader == null) {
                InputStream fileHeaderStream = this.getClass().getClassLoader()
                        .getResourceAsStream(CLASSIFIER_NAME + ".header");
                if(fileHeaderStream != null) {
                    Reader reader = new InputStreamReader(fileHeaderStream, "UTF-8");
                    trainHeader = new Instances(reader);
                    reader.close();
                }
                //DataSource source = new DataSource(CLASSIFIER_NAME + ".header");
                //Instances header = source.getDataSet();
                /*if (trainHeader.classIndex() == -1) {
                    trainHeader.setClassIndex(0);
                }*/
            }
        } catch (Exception e) {
            
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new TestSentimentEngine().init();
        String text = "Excellent pasta and meat sauce!"
                + "We decided to make a trip to North Beach after the lunch rush. It was easy to find parking on the street"
                + " just a block away. When we walked into the store, there was no line, but we were overwhelmed"
                + " with the various options on their menu. By the time we were ready to order, there was a group of people"
                + " perusing the menu as well. The store is neatly kept with various gourmet Italian pantry items."
                + " We asked the owner for a recommendation for the tortellini, which ended up hitting the spot."
                + " The pasta was cooked to perfection and the meat sauce was flavorful. We ended up splitting the tortellini,"
                + " which came out to the perfect proportion for two people."
                + " The service was great and they were quick it whipping up our order!"
                + " I definitely recommend this place :)";
        
        if (text != null && !text.isEmpty()) {
            String textArff = convertTextToARFF(text);
            try {
                Instances testInstances = loadFromArffString(textArff);
                Instances filteredTestInstance = convertToWordVectorFilter(testInstances);
                //System.out.println(filteredTestInstance);
                Instances mappedTestInstances = mapTestTrainAttributes(filteredTestInstance, trainHeader, false);
                mappedTestInstances.setClassIndex(0);

                double score = classifier.classifyInstance(mappedTestInstances.instance(0));
                String predicted = mappedTestInstances.classAttribute().value((int) score);
                
                //String predicted = filteredTestInstance.classAttribute().value((int) score);
                HashMap<String, String> result = new HashMap<String, String>();
                result.put("rating", predicted);

                Gson gson = new Gson();
                String jsonOut = gson.toJson(result);
                System.out.println(jsonOut);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }
    
    private static Instances convertToWordVectorFilter(Instances testInstances) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(testInstances);
        filter.setLowerCaseTokens(true);
        //filter.
        //filter.setUseStoplist(true);
        filter.setOutputWordCounts(true);
        filter.setWordsToKeep(100000);
        WordTokenizer wt = new WordTokenizer();
        String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
        wt.setDelimiters(delimiters);
        filter.setTokenizer(wt);
        //filter.setTFTransform(true);
        //filter.setStopwords(new File("stopwords.txt"));
        return Filter.useFilter(testInstances, filter);
    }

    private static String convertTextToARFF(String content) {
        String categoriesList = "1.0, 2.0, 3.0, 4.0, 5.0";

        String dataHeader =
                "@relation 'SentimentData'\n\n@attribute class-att {" + categoriesList
                        + "}\n@attribute Text string\n\n\n@data\n";
        String dummyCat = categoriesList.split(",", 2)[0];
        content = cleanContent(content);
        String testArff = "\"" + dummyCat + "\",\"" + content + "\"\n";

        return dataHeader + testArff;
    }

    private static String cleanContent(String content) {
        content = content.replaceAll("\n", " ");
        content = content.replaceAll("\"", "");
        content = content.replaceAll("\r", " ");
        content = content.replaceAll("\r\n", " ");
        content = content.replaceAll("\\\"","");
        return content;
    }
    
    static Instances loadFromArffString(String wekaData) throws IOException {
        Reader reader = new StringReader(wekaData);
        Instances instances = new Instances(reader);
        return instances;
    }

/*    public static Instance mapTestInstanceToTrain(Instance testData, Instances header ) {
        int numAttributes = header.numAttributes();
        double[] vals = new double[numAttributes];

        Map<String, Integer> testMap = new HashMap<String, Integer>();
        for (int i = 0; i < testData.numAttributes(); i++) {
            testMap.put(testData.attribute(i).name(), i);
        }

        for (int i = 0; i < numAttributes - 1; i++) {
            Attribute attribute = header.attribute(i);
            double value = Instance.missingValue();

            if (testMap.containsKey(attribute.name())) {
                int mapIndex = testMap.get(attribute.name());
                value = testData.value(mapIndex);
            }

            vals[i] = value;
        }

        Instance instance = new Instance(1.0, vals);
        instance.setDataset(header);
        return instance;
    }*/
    
    public static Instances mapTestTrainAttributes(Instances test,Instances trainData,boolean flag){    
        HashMap<String,Integer> testmap=new HashMap<String,Integer>();
        try {
            for(int i=0;i<test.numAttributes();i++){
                if(trainData.attribute(test.attribute(i).name())!=null){
                    testmap.put(test.attribute(i).name(),new Integer(i));
                }
            }
            
            //Gettting the class attribute nominals.
            String attrNominal= trainData.attribute(0).toString();
            String[] nominals=attrNominal.substring(attrNominal.indexOf("{")+1,attrNominal.indexOf("}")).split(",");
            
            StringBuffer out=new StringBuffer();            
            for(int i=0;i<test.numInstances();i++){
                Instance testInst=test.instance(i);             
                out.append("{");
                for(int j=0;j<trainData.numAttributes();j++){
                    Attribute attr=trainData.attribute(j);
                    if(testmap.containsKey(attr.name())){
                        int index=testmap.get(attr.name()).intValue();
                        int val=(int)testInst.value(index);
                        if(index==0 && flag) {                          
                            out.append(j+" "+nominals[val]+",");
                        }else if(val!=0 && index!=0) out.append(j+" "+val +",");
                    }
                }
                out.append("}\n");              
            }
            
            return loadFromArffString(trainData.toString()+"\n"+out.toString().replaceAll(",}", "}"));
            
        } catch (Exception e) {         
            e.printStackTrace();
        }
        return null;
    }

}
