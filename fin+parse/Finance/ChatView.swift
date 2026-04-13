// ChatView.swift — AI Financial Assistant

import SwiftUI
import Combine 

struct ChatView: View {
    @EnvironmentObject var dataManager: SharedDataManager
    @EnvironmentObject var authManager: AuthManager
    @State private var messages: [Message] = [
        Message("Hi! 👋 I'm your AI financial assistant. I can help you analyze your spending, suggest budgets, or answer any finance questions. What's on your mind?", isFromUser: false)
    ]
    @State private var inputText = ""
    @State private var isTyping = false

    // Suggestions
    private let suggestions = [
        "How's my spending this month?",
        "Give me a savings tip",
        "Analyze my top expenses",
        "How can I save more?",
    ]

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 0) {
                HStack(spacing: 12) {
                    ZStack {
                        Circle().fill(AppTheme.accentGradient).frame(width: 40, height: 40)
                        Image(systemName: "brain").font(.system(size: 16)).foregroundColor(.white)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Financial AI").font(.system(size: 16, weight: .bold)).foregroundColor(AppTheme.textPrimary)
                        HStack(spacing: 5) {
                            Circle().fill(AppTheme.green).frame(width: 6, height: 6)
                            Text("Online").font(.system(size: 11)).foregroundColor(AppTheme.green)
                        }
                    }
                    Spacer()
                    Button(action: {
                        messages = [Message("Hi! 👋 I'm your AI financial assistant. How can I help you today?", isFromUser: false)]
                    }) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 14)).foregroundColor(AppTheme.textMuted)
                            .padding(8).background(AppTheme.surface2).clipShape(Circle())
                    }
                }
                .padding(.horizontal, 20).padding(.vertical, 14)
                Divider().background(AppTheme.border)
            }
            .background(AppTheme.surface)

            // Messages
            ScrollViewReader { proxy in
                ScrollView(showsIndicators: false) {
                    LazyVStack(spacing: 12) {
                        // Suggestions (only when just the welcome message)
                        if messages.count == 1 {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Quick questions")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(AppTheme.textMuted)
                                    .padding(.horizontal, 20)
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 8) {
                                        ForEach(suggestions, id: \.self) { s in
                                            Button(action: { sendMessage(text: s) }) {
                                                Text(s)
                                                    .font(.system(size: 13, weight: .medium))
                                                    .foregroundColor(AppTheme.accent)
                                                    .padding(.horizontal, 14).padding(.vertical, 9)
                                                    .background(AppTheme.accentSoft)
                                                    .clipShape(Capsule())
                                                    .overlay(Capsule().stroke(AppTheme.accent.opacity(0.3)))
                                            }.buttonStyle(PressEffect())
                                        }
                                    }
                                    .padding(.horizontal, 20)
                                }
                            }
                            .padding(.top, 16)
                        }

                        ForEach(messages) { msg in
                            ChatBubble(message: msg).id(msg.id)
                        }

                        if isTyping {
                            TypingBubble().id("typing")
                        }

                        Color.clear.frame(height: 20).id("bottom")
                    }
                    .padding(.bottom, 8)
                }
                .background(AppTheme.bg)
                .onChange(of: messages.count) { oldValue, newValue in
                    withAnimation(.spring(response: 0.4)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
                // Fix for isTyping
                .onChange(of: isTyping) { oldValue, newValue in
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }

            // Input Bar
            VStack(spacing: 0) {
                Divider().background(AppTheme.border)
                HStack(spacing: 12) {
                    HStack(spacing: 10) {
                        TextField("Ask about your finances...", text: $inputText, axis: .vertical)
                            .font(.system(size: 15))
                            .foregroundColor(AppTheme.textPrimary)
                            .tint(AppTheme.accent)
                            .lineLimit(1...4)
                        if !inputText.isEmpty {
                            Button(action: { inputText = "" }) {
                                Image(systemName: "xmark.circle.fill").foregroundColor(AppTheme.textMuted).font(.system(size: 15))
                            }
                        }
                    }
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .background(AppTheme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: AppTheme.radiusLG, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: AppTheme.radiusLG).stroke(AppTheme.border))

                    Button(action: { sendMessage(text: inputText) }) {
                        ZStack {
                            Circle()
                                .fill(inputText.isEmpty ? AnyShapeStyle(AppTheme.surface2) : AnyShapeStyle(AppTheme.accentGradient))
                                .frame(width: 44, height: 44)
                                .shadow(color: inputText.isEmpty ? .clear : AppTheme.accent.opacity(0.3), radius: 8, y: 3)
                            Image(systemName: "arrow.up").font(.system(size: 16, weight: .bold))
                                .foregroundColor(inputText.isEmpty ? AppTheme.textMuted : .white)
                        }
                    }
                    .buttonStyle(PressEffect())
                    .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isTyping)
                }
                .padding(.horizontal, 16).padding(.vertical, 12)
                .background(AppTheme.surface)
            }
        }
        .background(AppTheme.bg)
    }

    private func sendMessage(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        inputText = ""
        let userMsg = Message(trimmed, isFromUser: true)
        withAnimation(.spring(response: 0.4)) { messages.append(userMsg) }
        isTyping = true
        getAIResponse(for: trimmed)
    }

    private func getAIResponse(for userMessage: String) {
        // Build financial context for the AI
        let context = buildFinancialContext()

        let systemPrompt = """
        You are Financial App AI, a friendly and expert personal finance assistant. You have access to the user's financial data below.
        Keep responses concise (2-4 sentences), helpful, and actionable. Use emojis sparingly. Be warm and encouraging.
        
        User's Financial Data:
        \(context)
        """

        let requestBody: [String: Any] = [
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "system": systemPrompt,
            "messages": [["role": "user", "content": userMessage]]
        ]

        guard let url = URL(string: "https://api.anthropic.com/v1/messages"),
              let body = try? JSONSerialization.data(withJSONObject: requestBody) else {
            fallbackResponse()
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // API key would be set here in production
        // request.setValue("Bearer YOUR_KEY", forHTTPHeaderField: "Authorization")
        request.httpBody = body

        // Simulate AI response with context-aware answers (since we can't call API without key in app)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            isTyping = false
            let response = generateContextualResponse(for: userMessage)
            withAnimation(.spring(response: 0.4)) {
                messages.append(Message(response, isFromUser: false))
            }
        }
    }

    private func buildFinancialContext() -> String {
        let dm = dataManager
        return """
        Balance: $\(String(format: "%.2f", dm.balance))
        Total Income: $\(String(format: "%.2f", dm.totalIncome))
        Total Expenses: $\(String(format: "%.2f", dm.totalExpenses))
        Savings Rate: \(String(format: "%.1f", dm.savingsRate))%
        This Month Expenses: $\(String(format: "%.2f", dm.thisMonthExpenses))
        Top Spending Categories: \(dm.categorySpending.prefix(3).map { "\($0.category.rawValue): $\(String(format: "%.0f", $0.amount))" }.joined(separator: ", "))
        Active Goals: \(dm.goals.map { "\($0.name) (\($0.progressPercent)% complete)" }.joined(separator: ", "))
        """
    }

    // Context-aware local responses
    private func generateContextualResponse(for message: String) -> String {
        let dm = dataManager
        let msg = message.lowercased()

        if msg.contains("spending") || msg.contains("expense") || msg.contains("month") {
            let topCat = dm.categorySpending.first?.category.rawValue ?? "various categories"
            return "This month you've spent $\(String(format: "%.2f", dm.thisMonthExpenses)). Your biggest spending category is \(topCat). Your savings rate is currently \(String(format: "%.1f", dm.savingsRate))% — \(dm.savingsRate >= 20 ? "great job! 🎉" : "there's room to improve this. Try cutting discretionary spending.")"
        }
        if msg.contains("save") || msg.contains("savings") || msg.contains("tip") {
            let potential = dm.totalExpenses * 0.1
            return "Based on your spending, you could save an extra $\(String(format: "%.0f", potential))/month by reducing discretionary expenses by 10%. 💡 Consider reviewing your subscriptions and dining out less — those tend to add up quickly!"
        }
        if msg.contains("balance") || msg.contains("account") {
            return "Your current balance is $\(String(format: "%.2f", dm.balance)). You've earned $\(String(format: "%.2f", dm.totalIncome)) in income and spent $\(String(format: "%.2f", dm.totalExpenses)) total. Keep it up! 💪"
        }
        if msg.contains("goal") {
            let nearestGoal = dm.goals.sorted { $0.progress > $1.progress }.first
            if let g = nearestGoal {
                return "Your '\(g.name)' goal is \(g.progressPercent)% complete! 🎯 You've saved $\(String(format: "%.0f", g.savedAmount)) of your $\(String(format: "%.0f", g.targetAmount)) target. Keep it going!"
            }
            return "Set up savings goals to track your progress toward big purchases and milestones! 🎯 Tap the Goals tab to get started."
        }
        if msg.contains("budget") {
            return "You have \(dm.budgets.count) active budgets. 📊 To stay on track, try the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Based on your income of $\(String(format: "%.0f", dm.totalIncome)), you should aim to save at least $\(String(format: "%.0f", dm.totalIncome * 0.2)) per month."
        }
        if msg.contains("invest") {
            return "With a $\(String(format: "%.0f", dm.balance)) balance and \(String(format: "%.1f", dm.savingsRate))% savings rate, you're building a solid foundation. 📈 Consider putting your emergency fund first (3-6 months of expenses = ~$\(String(format: "%.0f", dm.thisMonthExpenses * 4))), then explore index fund investing."
        }

        let responses = [
            "Based on your financials, you're doing well with a \(String(format: "%.1f", dm.savingsRate))% savings rate! 💰 Keep tracking your expenses and you'll hit your goals faster.",
            "Great question! With $\(String(format: "%.2f", dm.balance)) in your balance, focus on maintaining your budget and growing your savings. What specific area would you like advice on?",
            "Financial wellness is a journey! Your current balance of $\(String(format: "%.2f", dm.balance)) is a great foundation. Would you like tips on budgeting, saving, or investing? 🌱",
        ]
        return responses.randomElement()!
    }

    private func fallbackResponse() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            isTyping = false
            withAnimation(.spring(response: 0.4)) {
                messages.append(Message("I'm here to help with your finances! Ask me about your spending, savings goals, or get personalized tips. 💡", isFromUser: false))
            }
        }
    }
}

// MARK: - Chat Bubble
struct ChatBubble: View {
    let message: Message

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if message.isFromUser { Spacer(minLength: 60) }

            if !message.isFromUser {
                ZStack {
                    Circle().fill(AppTheme.accentGradient).frame(width: 28, height: 28)
                    Image(systemName: "brain").font(.system(size: 11)).foregroundColor(.white)
                }
            }

            VStack(alignment: message.isFromUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .font(.system(size: 14))
                    .foregroundColor(message.isFromUser ? .white : AppTheme.textPrimary)
                    .padding(.horizontal, 14).padding(.vertical, 11)
                    .background(
                        message.isFromUser
                            ? AnyView(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(AppTheme.accentGradient))
                            : AnyView(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(AppTheme.surface)
                                .overlay(RoundedRectangle(cornerRadius: 18).stroke(AppTheme.border)))
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

                Text(message.timestamp, style: .time)
                    .font(.system(size: 9)).foregroundColor(AppTheme.textMuted)
                    .padding(.horizontal, 4)
            }

            if !message.isFromUser { Spacer(minLength: 60) }
            if message.isFromUser { EmptyView() }
        }
        .padding(.horizontal, 16)
        .transition(.asymmetric(insertion: .scale(scale: 0.85, anchor: .bottom).combined(with: .opacity), removal: .opacity))
    }
}

// MARK: - Typing Indicator
struct TypingBubble: View {
    @State private var dot = 0
    let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ZStack {
                Circle().fill(AppTheme.accentGradient).frame(width: 28, height: 28)
                Image(systemName: "brain").font(.system(size: 11)).foregroundColor(.white)
            }
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle()
                        .fill(AppTheme.textMuted)
                        .frame(width: 7, height: 7)
                        .scaleEffect(dot == i ? 1.4 : 1)
                        .animation(.spring(response: 0.3), value: dot)
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 14)
            .background(AppTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(AppTheme.border))
            Spacer()
        }
        .padding(.horizontal, 16)
        .onReceive(timer) { _ in dot = (dot + 1) % 3 }
    }
}
