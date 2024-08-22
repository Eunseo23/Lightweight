package com.lightweight;

import java.io.*;
// import java.util.regex.*;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import com.github.difflib.DiffUtils;
import com.github.difflib.patch.AbstractDelta;
import com.github.difflib.patch.Patch;
import java.io.IOException;

public class OverallProcess {

     // 파일에서 모든 주석을 제거하는 메서드
     public static String removeCommentsFromFile(String inputFilePath) throws IOException {
        // 파일을 읽기 위해 BufferedReader 사용
        BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
        StringBuilder code = new StringBuilder();
        String line;

        // 파일의 모든 라인을 읽어들여 StringBuilder에 추가
        while ((line = reader.readLine()) != null) {
            code.append(line).append("\n");
        }
        reader.close();

        // 주석 제거
        return removeComments(code.toString());
    }

    // 주어진 자바 코드 문자열에서 주석을 제거하는 메서드
    public static String removeComments(String code) {
        String regex = "(?s)/\\*.*?\\*/|//.*?(\r?\n|$)";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(code);
        return matcher.replaceAll("").trim();
    }

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

    // 두 개의 파일에서 차이점을 계산하고 차이점이 발생한 메서드만 추출하는 메서드
    public static List<List<String>> extractModifiedMethodsFromFiles(String inputFilePath1, String inputFilePath2) {
        List<List<String>> extractedMethods = new ArrayList<>();
        
        try {
            // 첫 번째 파일에서 주석 제거 및 코드 처리
            String cleanedCode1 = RemoveComments.removeCommentsFromFile(inputFilePath1);
            String processedCode1 = JavaParserStatement.processCode(cleanedCode1);

            // 두 번째 파일에서 주석 제거 및 코드 처리
            String cleanedCode2 = RemoveComments.removeCommentsFromFile(inputFilePath2);
            String processedCode2 = JavaParserStatement.processCode(cleanedCode2);

            // 파일에서 텍스트 읽기
            List<String> original = List.of(processedCode1.split("\n"));
            List<String> revised = List.of(processedCode2.split("\n"));

            // 차이점 계산
            Patch<String> patch = DiffUtils.diff(original, revised);

            // 차이점이 발생한 메서드만 추출
            List<String> modifiedOriginalMethods = extractModifiedMethods(patch, original);
            List<String> modifiedRevisedMethods = extractModifiedMethods(patch, revised);

            // 결과 리스트에 추가
            extractedMethods.add(modifiedOriginalMethods);
            extractedMethods.add(modifiedRevisedMethods);
            
        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }

        return extractedMethods; // 두 리스트를 포함한 리스트 반환
    }
        // 메서드 추출 메서드 (메서드 내부 변경)
        private static List<String> extractModifiedMethods(Patch<String> patch, List<String> lines) {
            List<String> modifiedMethods = new ArrayList<>();
            Pattern methodPattern = Pattern.compile("^[\\w\\s<>\\[\\]]+\\s+\\w+\\s*\\([^\\)]*\\)\\s*\\{?");
            
            for (AbstractDelta<String> delta : patch.getDeltas()) {
                int startLine = delta.getTarget().getPosition();
                int methodStart = findMethodStart(lines, startLine, methodPattern);
                if (methodStart == -1) {
                    // 메서드 시작을 찾지 못한 경우, 해당 델타를 무시
                    continue;
                }
                
                int methodEnd = findMethodEnd(lines, methodStart);
                if (methodEnd != -1) {
                    // Method body 추출 및 List에 추가 (인덴트 제거)
                    List<String> methodLines = lines.subList(methodStart, methodEnd + 1);
                    modifiedMethods.addAll(methodLines);
                    modifiedMethods.add("");  // 메서드 간의 공백 줄 추가
                }
            }
        
            return modifiedMethods;
        }
    
    
        // 메서드 시작 위치 찾기
        private static int findMethodStart(List<String> lines, int startLine, Pattern methodPattern) {
            for (int i = startLine; i >= 0; i--) {
                Matcher matcher = methodPattern.matcher(lines.get(i).trim());
                if (matcher.find()) {
                    return i;
                }
            }
            return -1;
        }
    
        // 메서드 끝 위치 찾기
        private static int findMethodEnd(List<String> lines, int startLine) {
            int openBraces = 0;
            for (int i = startLine; i < lines.size(); i++) {
                String line = lines.get(i).trim();
                for (char c : line.toCharArray()) {
                    if (c == '{') openBraces++;
                    if (c == '}') openBraces--;
                }
                if (openBraces == 0 && line.endsWith("}")) {
                    return i;
                }
            }
            return lines.size() - 1;
        }

        public static void main(String[] args) {
        // 파일 경로 설정
        String inputFilePathp = "C:\\Users\\UOS\\Desktop\\data\\p_dir\\NotificationService.java"; // 첫 번째 파일
        String inputFilePathf = "C:\\Users\\UOS\\Desktop\\data\\f_dir\\NotificationService.java"; // 두 번째 파일
        String finalOriginalTextFilePath = "OriginalMethods.txt";
        String finalRevisedTextFilePath = "RevisedMethods.txt";

        try {
            // 1. ExtractBuggyMethod를 사용하여 파일 처리
            List<List<String>> modifiedMethods = ExtractBuggyMethod.extractModifiedMethodsFromFiles(inputFilePathp, inputFilePathf);
            List<String> modifiedOriginalMethods = modifiedMethods.get(0);
            List<String> modifiedRevisedMethods = modifiedMethods.get(1);

            // 빈 줄이 삽입된 결과를 저장할 리스트
            List<String> originalWithBlanks = new ArrayList<>(modifiedOriginalMethods);
            List<String> revisedWithBlanks = new ArrayList<>(modifiedRevisedMethods);

            // 차이점 계산 (원본과 수정본의 메서드에서 직접 비교)
            Patch<String> patch = DiffUtils.diff(originalWithBlanks, revisedWithBlanks);

            // 차이점 적용
            for (AbstractDelta<String> delta : patch.getDeltas()) {
                int originalPosition = delta.getSource().getPosition();
                int revisedPosition = delta.getTarget().getPosition();

                switch (delta.getType()) {
                    case DELETE:
                        // 삭제된 경우 revised에 빈 줄 삽입
                        revisedWithBlanks.add(revisedPosition, "");
                        break;
                    case INSERT:
                        // 추가된 경우 original에 빈 줄 삽입
                        originalWithBlanks.add(originalPosition, "");
                        break;
                    case CHANGE:
                    case EQUAL:
                    default:
                        break;
                }
            }

            // 2. 최종 텍스트 파일로 변환 및 저장

            // OriginalWithBlanks의 내용을 텍스트로 변환
            List<String> originalStrippedLines = originalWithBlanks.stream()
                .map(String::trim) // 앞뒤 공백 제거
                .collect(Collectors.toList());

            // 결과를 텍스트 파일로 저장
            Files.write(Paths.get(finalOriginalTextFilePath), originalStrippedLines);

            // RevisedWithBlanks의 내용을 텍스트로 변환
            List<String> revisedStrippedLines = revisedWithBlanks.stream()
                .map(String::trim) // 앞뒤 공백 제거
                .collect(Collectors.toList());

            // 결과를 텍스트 파일로 저장
            Files.write(Paths.get(finalRevisedTextFilePath), revisedStrippedLines);

            // 결과 출력
            // System.out.println("Difference applied and Java files converted to text files successfully.");
            // System.out.println("Saved as " + finalOriginalTextFilePath + " and " + finalRevisedTextFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}