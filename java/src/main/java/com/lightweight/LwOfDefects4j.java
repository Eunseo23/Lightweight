package com.lightweight;

import java.io.*;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.printer.configuration.PrettyPrinterConfiguration;
import com.github.javaparser.ast.Node;

import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import com.github.difflib.DiffUtils;
import com.github.difflib.patch.AbstractDelta;
import com.github.difflib.patch.Patch;

//statement단위로 나누고, 주석제거!!!!!!!!!!
public class LwOfDefects4j  {
    public static void main(String[] args) {
        String inputFilePath = "C:\\Users\\UOS\\Desktop\\LwD4j\\Closure\\44\\CodeConsumer.java"; // 주석을 제거할 자바 파일이 있음
        String outputFilePath = "C:\\Users\\UOS\\Desktop\\Lightweight\\Test.java"; // input한 파일명 + buggy method.txt로 저장
        try {
            // 주석 제거
            String codeWithoutComments = removeCommentsFromFile(inputFilePath);

            // 세미콜론을 기준으로 문장을 한 줄로 정리
            String processedCode = processCode(codeWithoutComments);

            // 처리된 결과의 인덴트 제거
            String codeWithoutIndentation = removeLeadingIndentation(processedCode);

            // 결과를 파일에 저장
            saveProcessedCode(outputFilePath, codeWithoutIndentation);

            System.out.println("Processed code has been saved to " + outputFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void saveProcessedCode(String outputFilePath, String processedCode) throws IOException {
        try (FileWriter writer = new FileWriter(outputFilePath)) {
            writer.write(processedCode);
        }
    }

    // 파일에서 모든 주석을 제거하는 메서드
    public static String removeCommentsFromFile(String inputFilePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
        StringBuilder code = new StringBuilder();
        String line;

        while ((line = reader.readLine()) != null) {
            code.append(line).append("\n");
        }
        reader.close();

        return removeComments(code.toString());
    }

    // 주어진 자바 코드 문자열에서 주석을 제거하는 메서드
    public static String removeComments(String code) {
        // 주석을 모두 제거하는 정규 표현식
        String regex = "(?s)/\\*.*?\\*/|//.*?(\r?\n|$)";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(code);
        return matcher.replaceAll("").trim();
    }

    // 코드에서 인덴트를 제거하는 메서드 (최종 단계로 이동)
    private static String removeLeadingIndentation(String code) {
        String[] lines = code.split("\n");
        StringBuilder result = new StringBuilder();

        // 모든 라인의 최소 인덴트 수준을 계산합니다.
        int minIndent = Integer.MAX_VALUE;
        for (String line : lines) {
            if (!line.trim().isEmpty()) { // 빈 줄은 무시합니다.
                int currentIndent = line.indexOf(line.trim());
                if (currentIndent < minIndent) {
                    minIndent = currentIndent;
                }
            }
        }

        // 최소 인덴트 수준만큼 인덴트를 제거합니다.
        for (String line : lines) {
            if (line.length() >= minIndent) {
                result.append(line.substring(minIndent)).append("\n");
            } else {
                result.append(line).append("\n");
            }
        }

        return result.toString().trim(); // 마지막에 trim()을 사용하여 불필요한 공백 제거
    }

    // 코드에서 표현식을 처리하여 세미콜론이 추가되도록 하고, 모든 문을 한 줄로 정리하는 메서드
    public static String processCode(String code) {
        try {
            JavaParser parser = new JavaParser();
            CompilationUnit cu = parser.parse(code).getResult().orElseThrow();

            // Modify the code to ensure each expression statement ends with a semicolon
            cu.accept(new VoidVisitorAdapter<Void>() {
                @Override
                public void visit(ExpressionStmt stmt, Void arg) {
                    super.visit(stmt, arg);
                    // Convert the statement to string and add a semicolon if necessary
                    String statementStr = stmt.toString().trim();
                    if (!statementStr.endsWith(";")) {
                        // Adding a semicolon to the statement
                        stmt.setExpression(stmt.getExpression().toString() + ";");
                    }
                }
            }, null);

            // Return the modified code as a string
            return cu.toString();

        } catch (Exception e) {
            e.printStackTrace();
            return code; // 예외 발생 시 원본 코드 반환
        }
    }
}

// 1 개 버기 메서드만 추출!!!!!!!!!
// public class LwOfDefects4j {

//         public static void main(String[] args) {
//             String javaFilePath = "C:/Users/UOS/Desktop/Lightweight/Partial_clean2.java"; // 자바 파일 경로

    
//             try {
//                 // 자바 파일에서 메서드를 추출
//                 List<String> methods = extractMethodsFromJavaFile(javaFilePath);
    
//                 // 패치 파일에서 +줄과 아무 기호 없는 줄을 추출
//                 Set<String> relevantLines = extractRelevantLinesFromPatch(patchFilePath);
    
//                 // 가장 많이 겹치는 메서드 찾기
//                 String bestMatchMethod = findBestMatchMethod(methods, relevantLines);
    
//                 // 코드의 앞에 있는 인덴트를 제거하고 저장
//                 if (bestMatchMethod != null) {
//                     bestMatchMethod = removeLeadingIndentation(bestMatchMethod);
//                     saveMethodToFile(outputFilePath, bestMatchMethod);
//                     System.out.println("Best matching method has been saved to " + outputFilePath);
//                 } else {
//                     System.out.println("No matching method found.");
//                 }
    
//             } catch (IOException e) {
//                 e.printStackTrace();
//             }
//         }
    
//         // 자바 파일에서 메서드를 추출하는 메서드
//         private static List<String> extractMethodsFromJavaFile(String javaFilePath) throws IOException {
//             List<String> methods = new ArrayList<>();
//             StringBuilder currentMethod = new StringBuilder();
//             boolean insideMethod = false;
//             int braceBalance = 0;
    
//             List<String> lines = Files.readAllLines(Paths.get(javaFilePath));
    
//             for (String line : lines) {
//                 String trimmedLine = line.trim();
    
//                 // 메서드 선언문을 감지 (public, private, protected, static으로 시작하는 줄)
//                 if (trimmedLine.matches("^(public|private|protected|static).*\\s*\\(.*\\)\\s*\\{?\\s*$")) {
//                     // 이전 메서드가 종료되지 않았다면, 메서드 추가
//                     if (insideMethod) {
//                         methods.add(currentMethod.toString());
//                         currentMethod.setLength(0);
//                     }
//                     insideMethod = true;
//                     braceBalance = 0; // 괄호 균형 초기화
//                 }
    
//                 if (insideMethod) {
//                     currentMethod.append(line).append("\n");
    
//                     // 중괄호 균형 맞추기
//                     braceBalance += countOccurrences(trimmedLine, '{');
//                     braceBalance -= countOccurrences(trimmedLine, '}');
    
//                     // 메서드가 끝났음을 감지 (괄호 균형이 맞을 때)
//                     if (braceBalance == 0 && trimmedLine.endsWith("}")) {
//                         methods.add(currentMethod.toString());
//                         currentMethod.setLength(0);
//                         insideMethod = false;
//                     }
//                 }
//             }
    
//             // 마지막 메서드를 추가 (파일의 끝에 도달했을 때)
//             if (insideMethod && currentMethod.length() > 0) {
//                 methods.add(currentMethod.toString());
//             }
    
//             return methods;
//         }
    
//         // 패치 파일에서 +줄과 아무 기호 없는 줄을 추출하는 메서드
//         private static Set<String> extractRelevantLinesFromPatch(String patchFilePath) throws IOException {
//             Set<String> relevantLines = new HashSet<>();
//             List<String> lines = Files.readAllLines(Paths.get(patchFilePath));
    
//             for (String line : lines) {
//                 if (line.startsWith("+") && !line.startsWith("+++")) {
//                     relevantLines.add(line.substring(1).trim()); // +를 제거하고 줄을 추가
//                 } else if (!line.startsWith("-") && !line.startsWith("@") && !line.startsWith("+++")) {
//                     relevantLines.add(line.trim()); // 그대로 줄을 추가
//                 }
//             }
    
//             return relevantLines;
//         }
    
//         // 가장 많이 겹치는 메서드를 찾는 메서드
//         private static String findBestMatchMethod(List<String> methods, Set<String> relevantLines) {
//             String bestMatchMethod = null;
//             int maxMatchCount = 0;
    
//             for (String method : methods) {
//                 int matchCount = countMatchesInMethod(method, relevantLines);
    
//                 if (matchCount > maxMatchCount) {
//                     maxMatchCount = matchCount;
//                     bestMatchMethod = method;
//                 }
//             }
    
//             return bestMatchMethod;
//         }
    
//         // 메서드 내에서 relevantLines와 일치하는 줄의 수를 세는 메서드
//         private static int countMatchesInMethod(String method, Set<String> relevantLines) {
//             int matchCount = 0;
    
//             for (String relevantLine : relevantLines) {
//                 if (method.contains(relevantLine)) {
//                     matchCount++;
//                 }
//             }
    
//             return matchCount;
//         }
    
//         // 추출된 메서드를 파일에 저장하는 메서드
//         private static void saveMethodToFile(String outputFilePath, String method) throws IOException {
//             try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
//                 writer.write(method);
//             }
//         }
    
//         // 코드 앞에 있는 인덴트 (스페이스, 탭) 제거하는 메서드
//         private static String removeLeadingIndentation(String code) {
//             String[] lines = code.split("\n");
//             StringBuilder result = new StringBuilder();
    
//             for (String line : lines) {
//                 result.append(line.stripLeading()).append("\n");  // 각 줄의 앞에 있는 공백 제거
//             }
    
//             return result.toString();
//         }
    
//         // 주어진 문자열에서 특정 문자의 등장 횟수를 세는 메서드
//         private static int countOccurrences(String str, char character) {
//             int count = 0;
//             for (char c : str.toCharArray()) {
//                 if (c == character) {
//                     count++;
//                 }
//             }
//             return count;
//         }
//     }

//폴더 전체 적용하기
// public class LwOfDefects4j {
// public static void main(String[] args) throws IOException {
//     String baseDirPath = "C:/Users/UOS/Desktop/LwD4j/Mockito/";
//     String patchBaseDirPath = "C:/Users/UOS/Desktop/MCRepair-main/APR_Resources/localization/defects4j_developers/Mockito/patches/";

//     // Get all directories in the base directory
//     List<String> subDirNames = getSubDirectoryNames(baseDirPath);

//     for (String dirName : subDirNames) {
//         int dirNumber;
//         try {
//             dirNumber = Integer.parseInt(dirName);
//         } catch (NumberFormatException e) {
//             // If the directory name is not a number, skip it
//             continue;
//         }

//         String inputDirPath = baseDirPath + dirName + "/";
//         String patchFilePath = patchBaseDirPath + dirNumber + ".src.patch";
//         String outputDirPath = inputDirPath;

//         System.out.println("Processing directory: " + dirName);

//         try {
//             // Get all Java files in the input directory
//             List<File> javaFiles = getJavaFilesInDirectory(inputDirPath);

//             // Get all relevant lines for each diff section from the patch file
//             List<Map<String, List<Set<String>>>> allRelevantLinesForFiles = extractAllRelevantLinesFromPatch(patchFilePath);

//             int methodCount = 1;  // For naming the output files

//             // Process each file with its corresponding diff sections
//             for (int i = 0; i < javaFiles.size(); i++) {
//                 File javaFile = javaFiles.get(i);
//                 Map<String, List<Set<String>>> relevantLinesMap = allRelevantLinesForFiles.get(i);

//                 // 주석 제거 및 코드 전처리
//                 String codeWithoutComments = removeCommentsFromFile(javaFile.getAbsolutePath());
//                 String processedCode = processCode(codeWithoutComments);

//                 // 자바 파일에서 메서드를 추출
//                 List<String> methods = extractMethodsFromCode(processedCode);

//                 // 각 diff 섹션에 대해 메서드 추출
//                 for (Map.Entry<String, List<Set<String>>> entry : relevantLinesMap.entrySet()) {
//                     String fileName = entry.getKey();
//                     List<Set<String>> relevantLinesList = entry.getValue();

//                     for (int sectionCount = 0; sectionCount < relevantLinesList.size(); sectionCount++) {
//                         Set<String> relevantLines = relevantLinesList.get(sectionCount);
//                         List<String> bestMatchMethods = findBestMatchMethods(methods, relevantLines);

//                         // 각 메서드 저장
//                         if (!bestMatchMethods.isEmpty()) {
//                             saveMethodsToFiles(outputDirPath, fileName, bestMatchMethods, methodCount);
//                             methodCount += bestMatchMethods.size();
//                             System.out.println("Best matching methods have been saved to " + outputDirPath);
//                         } else {
//                             System.out.println("No matching methods found in " + javaFile.getName());
//                         }
//                     }
//                 }
//             }
//         } catch (Exception e) {
//             // Log and skip to the next directory if an error occurs
//             System.err.println("Error processing directory " + dirName + ": " + e.getMessage());
//             e.printStackTrace();
//             continue;
//         }
//     }
// }

// private static List<String> getSubDirectoryNames(String baseDirPath) {
//     List<String> subDirNames = new ArrayList<>();
//     try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(baseDirPath))) {
//         for (Path entry : stream) {
//             if (Files.isDirectory(entry)) {
//                 subDirNames.add(entry.getFileName().toString());
//             }
//         }
//     } catch (IOException e) {
//         e.printStackTrace();
//     }
//     return subDirNames;
// }

// // 특정 디렉토리에서 모든 .java 파일을 가져오는 메서드
// private static List<File> getJavaFilesInDirectory(String dirPath) {
//     List<File> javaFiles = new ArrayList<>();
//     try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(dirPath), "*.java")) {
//         for (Path entry : stream) {
//             javaFiles.add(entry.toFile());
//         }
//     } catch (IOException e) {
//         e.printStackTrace();
//     }
//     return javaFiles;
// }

// // 여러 메서드를 파일로 저장하는 메서드
// private static void saveMethodsToFiles(String outputDirPath, String fileName, List<String> methods, int startCount) throws IOException {
//     for (int i = 0; i < methods.size(); i++) {
//         String outputFilePath = outputDirPath + fileName + "_buggymethod_" + (startCount + i) + ".txt";
//         String methodWithNoIndentation = removeLeadingIndentation(methods.get(i));
//         try (FileWriter writer = new FileWriter(outputFilePath)) {
//             writer.write(methodWithNoIndentation);
//         }
//     }
// }

// public static String removeCommentsFromFile(String inputFilePath) throws IOException {
//     BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
//     StringBuilder code = new StringBuilder();
//     String line;

//     while ((line = reader.readLine()) != null) {
//         code.append(line).append("\n");
//     }
//     reader.close();

//     return removeComments(code.toString());
// }

// public static String removeComments(String code) {
//     String regex = "(?s)/\\*.*?\\*/|//.*?(\r?\n|$)";
//     Pattern pattern = Pattern.compile(regex);
//     Matcher matcher = pattern.matcher(code);
//     return matcher.replaceAll("").trim();
// }

// public static String processCode(String code) {
//     try {
//         JavaParser parser = new JavaParser();
//         CompilationUnit cu = parser.parse(code).getResult().orElseThrow();

//         cu.accept(new VoidVisitorAdapter<Void>() {
//             @Override
//             public void visit(ExpressionStmt stmt, Void arg) {
//                 super.visit(stmt, arg);
//                 String statementStr = stmt.toString().trim();
//                 if (!statementStr.endsWith(";")) {
//                     stmt.setExpression(stmt.getExpression().toString() + ";");
//                 }
//             }
//         }, null);

//         return cu.toString();

//     } catch (Exception e) {
//         e.printStackTrace();
//         return code;
//     }
// }

// // 코드 전체에서 메서드를 추출하는 메서드
// private static List<String> extractMethodsFromCode(String code) {
//     List<String> methods = new ArrayList<>();
//     StringBuilder currentMethod = new StringBuilder();
//     boolean insideMethod = false;
//     int braceBalance = 0;

//     String[] lines = code.split("\n");

//     for (String line : lines) {
//         String trimmedLine = line.trim();

//         if (trimmedLine.matches("^(public|private|protected|static).*\\s*\\(.*\\)\\s*\\{?\\s*$")) {
//             if (insideMethod) {
//                 methods.add(currentMethod.toString());
//                 currentMethod.setLength(0);
//             }
//             insideMethod = true;
//             braceBalance = 0;
//         }

//         if (insideMethod) {
//             currentMethod.append(line).append("\n");

//             braceBalance += countOccurrences(trimmedLine, '{');
//             braceBalance -= countOccurrences(trimmedLine, '}');

//             if (braceBalance == 0 && trimmedLine.endsWith("}")) {
//                 methods.add(currentMethod.toString());
//                 currentMethod.setLength(0);
//                 insideMethod = false;
//             }
//         }
//     }

//     if (insideMethod && currentMethod.length() > 0) {
//         methods.add(currentMethod.toString());
//     }

//     return methods;
// }

// // 패치 파일에서 각 diff 섹션의 +줄과 아무 기호 없는 줄을 추출하는 메서드
// private static List<Map<String, List<Set<String>>>> extractAllRelevantLinesFromPatch(String patchFilePath) throws IOException {
//     List<Map<String, List<Set<String>>>> allRelevantLinesForFiles = new ArrayList<>();
//     Map<String, List<Set<String>>> currentFileMap = null;
//     List<Set<String>> currentSections = null;
//     Set<String> currentRelevantLines = new HashSet<>();
//     String currentFileName = null;
//     List<String> lines = Files.readAllLines(Paths.get(patchFilePath));

//     for (String line : lines) {
//         if (line.startsWith("diff --git")) {
//             if (currentFileMap != null && currentFileName != null) {
//                 if (!currentRelevantLines.isEmpty()) {
//                     currentSections.add(new HashSet<>(currentRelevantLines));
//                 }
//                 currentFileMap.put(currentFileName, currentSections);
//                 allRelevantLinesForFiles.add(currentFileMap);
//             }
//             currentFileMap = new HashMap<>();
//             currentSections = new ArrayList<>();
//             currentFileName = extractFileName(line);
//             currentRelevantLines.clear();
//         } else if (line.startsWith("@@")) {
//             if (!currentRelevantLines.isEmpty()) {
//                 currentSections.add(new HashSet<>(currentRelevantLines));
//                 currentRelevantLines.clear();
//             }
//         } else if (line.startsWith("+") && !line.startsWith("+++")) {
//             currentRelevantLines.add(line.substring(1).trim());
//         } else if (!line.startsWith("-") && !line.startsWith("@") && !line.startsWith("+++")) {
//             currentRelevantLines.add(line.trim());
//         }
//     }

//     if (currentFileMap != null && currentFileName != null) {
//         if (!currentRelevantLines.isEmpty()) {
//             currentSections.add(currentRelevantLines);
//         }
//         currentFileMap.put(currentFileName, currentSections);
//         allRelevantLinesForFiles.add(currentFileMap);
//     }

//     return allRelevantLinesForFiles;
// }

// private static String extractFileName(String diffLine) {
//     String[] parts = diffLine.split(" ");
//     for (String part : parts) {
//         if (part.endsWith(".java")) {
//             return Paths.get(part).getFileName().toString().replace(".java", "");
//         }
//     }
//     return "unknown";
// }

// private static List<String> findBestMatchMethods(List<String> methods, Set<String> relevantLines) {
//     List<String> bestMatchMethods = new ArrayList<>();
//     int maxMatchCount = 0;

//     for (String method : methods) {
//         int matchCount = countMatchesInMethod(method, relevantLines);

//         if (matchCount > maxMatchCount) {
//             bestMatchMethods.clear();
//             bestMatchMethods.add(method);
//             maxMatchCount = matchCount;
//         } else if (matchCount == maxMatchCount && matchCount > 0) {
//             bestMatchMethods.add(method);
//         }
//     }

//     return bestMatchMethods;
// }

// private static int countMatchesInMethod(String method, Set<String> relevantLines) {
//     int matchCount = 0;

//     for (String relevantLine : relevantLines) {
//         if (method.contains(relevantLine)) {
//             matchCount++;
//         }
//     }

//     return matchCount;
// }

// private static String removeLeadingIndentation(String code) {
//     String[] lines = code.split("\n");
//     StringBuilder result = new StringBuilder();

//     for (String line : lines) {
//         result.append(line.stripLeading()).append("\n");
//     }

//     return result.toString();
// }

// private static int countOccurrences(String str, char character) {
//     int count = 0;
//     for (char c : str.toCharArray()) {
//         if (c == character) {
//             count++;
//         }
//     }
//     return count;
// }
// }