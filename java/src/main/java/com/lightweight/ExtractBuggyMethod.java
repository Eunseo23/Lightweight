package com.lightweight;

import java.io.IOException;
// import java.nio.file.Files;
// import java.nio.file.Paths;
// import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.github.difflib.DiffUtils;
import com.github.difflib.patch.AbstractDelta;
// import com.github.difflib.patch.DeltaType;
import com.github.difflib.patch.Patch;

public class ExtractBuggyMethod {

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
            Map<Integer, Integer> processedMethods = new HashMap<>();

            for (AbstractDelta<String> delta : patch.getDeltas()) {
                int startLine = delta.getTarget().getPosition();
                int methodStart = findMethodStart(lines, startLine, methodPattern);
                if (methodStart == -1) continue;

                int methodEnd = findMethodEnd(lines, methodStart);
                if (methodEnd == -1) continue;

                if (!processedMethods.containsKey(methodStart) || processedMethods.get(methodStart) < methodEnd) {
                    processedMethods.put(methodStart, methodEnd);
                }
            }

            for (Map.Entry<Integer, Integer> entry : processedMethods.entrySet()) {
                int start = entry.getKey();
                int end = entry.getValue();
                modifiedMethods.addAll(lines.subList(start, end + 1));
                modifiedMethods.add("");
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
    }