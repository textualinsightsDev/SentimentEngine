package com.textualinsights.mirchiweb;

import com.google.gson.Gson;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@SuppressWarnings("serial")
public class SentimentExtractionServlet extends HttpServlet {
    private static final String TEXT_PARAMETER = "text";
    private static final String CLASSIFIER_NAME = "SentimentEngine";
    private static Classifier classifier = null;
    private static Instances trainHeader = null;
    
    public void init()  throws ServletException {
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
            if(classifier == null) {
                log("classifier not getting loaded");
            }
            if (trainHeader == null) {
                InputStream fileHeaderStream = this.getClass().getClassLoader()
                        .getResourceAsStream(CLASSIFIER_NAME + ".header");
                if(fileHeaderStream != null) {
                    Reader reader = new InputStreamReader(fileHeaderStream, "UTF-8");
                    trainHeader = new Instances(reader);
                    reader.close();
                    trainHeader.setClassIndex(0);
                }
            }
            if (classifier == null || trainHeader == null ) {
                log("init servlet failed: ");
            }
        } catch (Exception e) {
            e.printStackTrace();
            log("init servlet failed: "+e.getMessage());
            throw new ServletException(e);
        }
    }
    
    protected void doGet(HttpServletRequest req, HttpServletResponse res) {
        serviceRequest(req, res);
    }
    
    protected void doPost(HttpServletRequest req, HttpServletResponse res) {
        serviceRequest(req, res);
    }

    private void serviceRequest(HttpServletRequest req, HttpServletResponse res) {
        String text = req.getParameter(TEXT_PARAMETER);
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
                res.setContentType("application/json; charset=utf-8");
                PrintWriter writer = res.getWriter();
                writer.print(jsonOut);
                
            } catch (IOException e) {
                log("Unable to create test ARFF. \n"+e);
                e.printStackTrace();
            } catch (Exception e) {
                log("Unable to classify. \n"+e);
                e.printStackTrace();
            }
        }
    }
    
    private String convertTextToARFF(String content) {
        String categoriesList = "1.0,2.0,3.0,4.0,5.0";
        String dataHeader =
                "@relation 'SentimentData'\n@attribute class-att {" + categoriesList
                        + "}\n@attribute text string\n\n@data\n";
        String dummyCat = categoriesList.split(",", 2)[0];
        content = cleanContent(content);
        String testArff = "\"" + dummyCat + "\",\"" + content + "\"\n";

        return dataHeader + testArff;
    }

    private String cleanContent(String content) {
        content = content.replaceAll("\n", " ");
        content = content.replaceAll("\"", "");
        content = content.replaceAll("\r", " ");
        content = content.replaceAll("\r\n", " ");
        content = content.replaceAll("\\\"","");
        return content;
    }
    
    private Instances loadFromArffString(String wekaData) throws IOException {
        Reader reader = new StringReader(wekaData);
        Instances instances = new Instances(reader);
        return instances;
    }
    
    private Instances mapTestTrainAttributes(Instances test,Instances trainData,boolean flag){    
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
    
    private Instances convertToWordVectorFilter(Instances testInstances) throws Exception {
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


}