// LocalizationManager.swift — Multi-language support (English, Russian, Kazakh)

import SwiftUI
import Combine

// MARK: - Supported Languages
enum AppLanguage: String, CaseIterable, Codable {
    case english = "en"
    case russian = "ru"
    case kazakh  = "kk"

    var displayName: String {
        switch self {
        case .english: return "English"
        case .russian: return "Русский"
        case .kazakh:  return "Қазақша"
        }
    }
    var flag: String {
        switch self {
        case .english: return "🇺🇸"
        case .russian: return "🇷🇺"
        case .kazakh:  return "🇰🇿"
        }
    }
}

// MARK: - Localization Manager
class LocalizationManager: ObservableObject {
    @Published var language: AppLanguage {
        didSet {
            UserDefaults.standard.set(language.rawValue, forKey: "fincora_language")
        }
    }

    init() {
        let saved = UserDefaults.standard.string(forKey: "fincora_language") ?? ""
        self.language = AppLanguage(rawValue: saved) ?? .english
    }

    func str(_ key: LocalizedKey) -> String {
        return key.value(for: language)
    }
}

// MARK: - Environment Key
private struct LocalizationManagerKey: EnvironmentKey {
    static let defaultValue = LocalizationManager()
}
extension EnvironmentValues {
    var localization: LocalizationManager {
        get { self[LocalizationManagerKey.self] }
        set { self[LocalizationManagerKey.self] = newValue }
    }
}

// MARK: - Localized Key Enum
enum LocalizedKey {
    // MARK: General
    case appName
    case cancel
    case save
    case delete
    case done
    case ok
    case close
    case edit
    case add
    case seeAll
    case signOut
    case loading

    // MARK: Auth
    case welcomeBack
    case signInContinue
    case email
    case password
    case fullName
    case confirmPassword
    case login
    case register
    case createAccount
    case alreadyHaveAccount
    case dontHaveAccount
    case forgotPassword

    // MARK: Dashboard
    case goodMorning
    case goodAfternoon
    case goodEvening
    case totalBalance
    case thisMonth
    case income
    case expenses
    case savings
    case cashFlow
    case incomeVsExpenses
    case spendingBreakdown
    case recentTransactions
    case noExpensesYet

    // MARK: Transactions
    case transactions
    case allTransactions
    case addTransaction
    case filterBy
    case searchTransactions
    case noTransactions
    case title
    case amount
    case category
    case date
    case type
    case expense
    case incomeLabel
    case note
    case saveTransaction
    case deleteTransaction
    case editTransaction

    // MARK: Budget
    case budget
    case stayOnTrack
    case budgetVsActual
    case monthlySpending
    case budgetLabel
    case spent
    case left
    case categories
    case setBudget
    case monthlyLimit
    case saveBudget

    // MARK: Goals
    case goals
    case dreamBig
    case progressOverview
    case noGoals
    case noGoalsDesc
    case createGoal
    case newGoal
    case pickEmoji
    case colorLabel
    case goalName
    case goalNamePlaceholder
    case targetAmount
    case savedAmount
    case deadline
    case goalAchieved
    case moreToGoal

    // MARK: Chat
    case aiAdvisor
    case askAnything
    case typeMessage

    // MARK: Account
    case account
    case profile
    case editName
    case enterNewName
    case notifications
    case budgetAlerts
    case weeklyReport
    case appearance
    case language
    case darkMode
    case memberSince
    case totalTransactions
    case signOutConfirm
    case signOutMessage
    case deleteAccount
    case deleteAccountConfirm
    case deleteAccountMessage
    case dangerZone
    case preferences
    case support
    case helpCenter
    case privacyPolicy
    case version

    // MARK: Bank Statement
    case uploadStatement
    case importTransactions
    case selectFile
    case parsing
    case reviewTransactions
    case importSelected
    case selectAll
    case deselectAll

    // MARK: Error Messages
    case emailTaken
    case invalidCredentials
    case weakPassword
    case emptyFields
    case passwordMismatch

    func value(for language: AppLanguage) -> String {
        switch language {
        case .english: return english
        case .russian: return russian
        case .kazakh:  return kazakh
        }
    }

    // MARK: - English
    var english: String {
        switch self {
        case .appName: return "Fincora"
        case .cancel: return "Cancel"
        case .save: return "Save"
        case .delete: return "Delete"
        case .done: return "Done"
        case .ok: return "OK"
        case .close: return "Close"
        case .edit: return "Edit"
        case .add: return "Add"
        case .seeAll: return "See all"
        case .signOut: return "Sign Out"
        case .loading: return "Loading..."

        case .welcomeBack: return "Welcome back"
        case .signInContinue: return "Sign in to continue"
        case .email: return "Email"
        case .password: return "Password"
        case .fullName: return "Full Name"
        case .confirmPassword: return "Confirm Password"
        case .login: return "Sign In"
        case .register: return "Sign Up"
        case .createAccount: return "Create Account"
        case .alreadyHaveAccount: return "Already have an account?"
        case .dontHaveAccount: return "Don't have an account?"
        case .forgotPassword: return "Forgot password?"

        case .goodMorning: return "Good morning"
        case .goodAfternoon: return "Good afternoon"
        case .goodEvening: return "Good evening"
        case .totalBalance: return "Total Balance"
        case .thisMonth: return "This Month"
        case .income: return "Income"
        case .expenses: return "Expenses"
        case .savings: return "Savings"
        case .cashFlow: return "Cash Flow"
        case .incomeVsExpenses: return "Income vs Expenses"
        case .spendingBreakdown: return "Spending Breakdown"
        case .recentTransactions: return "Recent Transactions"
        case .noExpensesYet: return "No expenses yet — start adding transactions!"

        case .transactions: return "Transactions"
        case .allTransactions: return "All Transactions"
        case .addTransaction: return "Add Transaction"
        case .filterBy: return "Filter"
        case .searchTransactions: return "Search transactions..."
        case .noTransactions: return "No transactions found"
        case .title: return "Title"
        case .amount: return "Amount"
        case .category: return "Category"
        case .date: return "Date"
        case .type: return "Type"
        case .expense: return "Expense"
        case .incomeLabel: return "Income"
        case .note: return "Note"
        case .saveTransaction: return "Save Transaction"
        case .deleteTransaction: return "Delete Transaction"
        case .editTransaction: return "Edit Transaction"

        case .budget: return "Budget"
        case .stayOnTrack: return "Stay on track with spending limits"
        case .budgetVsActual: return "Budget vs Actual"
        case .monthlySpending: return "Monthly spending by category"
        case .budgetLabel: return "Budget"
        case .spent: return "Spent"
        case .left: return "Left"
        case .categories: return "Categories"
        case .setBudget: return "Set Budget"
        case .monthlyLimit: return "Monthly Limit"
        case .saveBudget: return "Save Budget"

        case .goals: return "Goals"
        case .dreamBig: return "Dream big, save smart"
        case .progressOverview: return "Progress Overview"
        case .noGoals: return "No goals yet"
        case .noGoalsDesc: return "Set your first savings goal and start working towards your dreams!"
        case .createGoal: return "Create a Goal"
        case .newGoal: return "New Goal"
        case .pickEmoji: return "Pick an Emoji"
        case .colorLabel: return "Color"
        case .goalName: return "Goal Name"
        case .goalNamePlaceholder: return "e.g. Emergency Fund"
        case .targetAmount: return "Target ($)"
        case .savedAmount: return "Saved ($)"
        case .deadline: return "Deadline"
        case .goalAchieved: return "Goal achieved! 🎉"
        case .moreToGoal: return "more to reach your goal"

        case .aiAdvisor: return "AI Advisor"
        case .askAnything: return "Ask me anything about your finances"
        case .typeMessage: return "Type a message..."

        case .account: return "Account"
        case .profile: return "Profile"
        case .editName: return "Edit Name"
        case .enterNewName: return "Enter your new display name."
        case .notifications: return "Notifications"
        case .budgetAlerts: return "Budget Alerts"
        case .weeklyReport: return "Weekly Report"
        case .appearance: return "Appearance"
        case .language: return "Language"
        case .darkMode: return "Dark Mode"
        case .memberSince: return "Member since"
        case .totalTransactions: return "Total transactions"
        case .signOutConfirm: return "Sign Out?"
        case .signOutMessage: return "You'll need to sign in again to access your data."
        case .deleteAccount: return "Delete Account"
        case .deleteAccountConfirm: return "Delete Account?"
        case .deleteAccountMessage: return "This will permanently delete your account and all data. This cannot be undone."
        case .dangerZone: return "Danger Zone"
        case .preferences: return "Preferences"
        case .support: return "Support"
        case .helpCenter: return "Help Center"
        case .privacyPolicy: return "Privacy Policy"
        case .version: return "Version"

        case .uploadStatement: return "Upload Statement"
        case .importTransactions: return "Import Transactions"
        case .selectFile: return "Select PDF File"
        case .parsing: return "Parsing your statement..."
        case .reviewTransactions: return "Review Transactions"
        case .importSelected: return "Import Selected"
        case .selectAll: return "Select All"
        case .deselectAll: return "Deselect All"

        case .emailTaken: return "An account with this email already exists."
        case .invalidCredentials: return "Incorrect email or password."
        case .weakPassword: return "Password must be at least 8 characters."
        case .emptyFields: return "Please fill in all fields."
        case .passwordMismatch: return "Passwords do not match."
        }
    }

    // MARK: - Russian
    var russian: String {
        switch self {
        case .appName: return "Fincora"
        case .cancel: return "Отмена"
        case .save: return "Сохранить"
        case .delete: return "Удалить"
        case .done: return "Готово"
        case .ok: return "ОК"
        case .close: return "Закрыть"
        case .edit: return "Изменить"
        case .add: return "Добавить"
        case .seeAll: return "Все"
        case .signOut: return "Выйти"
        case .loading: return "Загрузка..."

        case .welcomeBack: return "С возвращением"
        case .signInContinue: return "Войдите, чтобы продолжить"
        case .email: return "Email"
        case .password: return "Пароль"
        case .fullName: return "Полное имя"
        case .confirmPassword: return "Подтвердите пароль"
        case .login: return "Войти"
        case .register: return "Зарегистрироваться"
        case .createAccount: return "Создать аккаунт"
        case .alreadyHaveAccount: return "Уже есть аккаунт?"
        case .dontHaveAccount: return "Нет аккаунта?"
        case .forgotPassword: return "Забыли пароль?"

        case .goodMorning: return "Доброе утро"
        case .goodAfternoon: return "Добрый день"
        case .goodEvening: return "Добрый вечер"
        case .totalBalance: return "Общий баланс"
        case .thisMonth: return "Этот месяц"
        case .income: return "Доходы"
        case .expenses: return "Расходы"
        case .savings: return "Сбережения"
        case .cashFlow: return "Денежный поток"
        case .incomeVsExpenses: return "Доходы и расходы"
        case .spendingBreakdown: return "Структура расходов"
        case .recentTransactions: return "Последние транзакции"
        case .noExpensesYet: return "Нет расходов — начните добавлять транзакции!"

        case .transactions: return "Транзакции"
        case .allTransactions: return "Все транзакции"
        case .addTransaction: return "Добавить транзакцию"
        case .filterBy: return "Фильтр"
        case .searchTransactions: return "Поиск транзакций..."
        case .noTransactions: return "Транзакции не найдены"
        case .title: return "Название"
        case .amount: return "Сумма"
        case .category: return "Категория"
        case .date: return "Дата"
        case .type: return "Тип"
        case .expense: return "Расход"
        case .incomeLabel: return "Доход"
        case .note: return "Заметка"
        case .saveTransaction: return "Сохранить транзакцию"
        case .deleteTransaction: return "Удалить транзакцию"
        case .editTransaction: return "Изменить транзакцию"

        case .budget: return "Бюджет"
        case .stayOnTrack: return "Контролируйте свои расходы"
        case .budgetVsActual: return "Бюджет и факт"
        case .monthlySpending: return "Расходы по категориям за месяц"
        case .budgetLabel: return "Бюджет"
        case .spent: return "Потрачено"
        case .left: return "Осталось"
        case .categories: return "Категории"
        case .setBudget: return "Установить бюджет"
        case .monthlyLimit: return "Месячный лимит"
        case .saveBudget: return "Сохранить бюджет"

        case .goals: return "Цели"
        case .dreamBig: return "Мечтай смело, копи мудро"
        case .progressOverview: return "Обзор прогресса"
        case .noGoals: return "Целей пока нет"
        case .noGoalsDesc: return "Поставьте первую цель и начните путь к своей мечте!"
        case .createGoal: return "Создать цель"
        case .newGoal: return "Новая цель"
        case .pickEmoji: return "Выберите эмодзи"
        case .colorLabel: return "Цвет"
        case .goalName: return "Название цели"
        case .goalNamePlaceholder: return "Например, Резервный фонд"
        case .targetAmount: return "Цель (₸)"
        case .savedAmount: return "Накоплено (₸)"
        case .deadline: return "Срок"
        case .goalAchieved: return "Цель достигнута! 🎉"
        case .moreToGoal: return "ещё до достижения цели"

        case .aiAdvisor: return "ИИ-советник"
        case .askAnything: return "Спросите меня о своих финансах"
        case .typeMessage: return "Написать сообщение..."

        case .account: return "Аккаунт"
        case .profile: return "Профиль"
        case .editName: return "Изменить имя"
        case .enterNewName: return "Введите новое отображаемое имя."
        case .notifications: return "Уведомления"
        case .budgetAlerts: return "Оповещения о бюджете"
        case .weeklyReport: return "Еженедельный отчёт"
        case .appearance: return "Внешний вид"
        case .language: return "Язык"
        case .darkMode: return "Тёмный режим"
        case .memberSince: return "Участник с"
        case .totalTransactions: return "Всего транзакций"
        case .signOutConfirm: return "Выйти?"
        case .signOutMessage: return "Вам нужно будет войти снова для доступа к данным."
        case .deleteAccount: return "Удалить аккаунт"
        case .deleteAccountConfirm: return "Удалить аккаунт?"
        case .deleteAccountMessage: return "Это навсегда удалит ваш аккаунт и все данные. Действие необратимо."
        case .dangerZone: return "Опасная зона"
        case .preferences: return "Настройки"
        case .support: return "Поддержка"
        case .helpCenter: return "Центр помощи"
        case .privacyPolicy: return "Политика конфиденциальности"
        case .version: return "Версия"

        case .uploadStatement: return "Загрузить выписку"
        case .importTransactions: return "Импорт транзакций"
        case .selectFile: return "Выбрать PDF файл"
        case .parsing: return "Обработка выписки..."
        case .reviewTransactions: return "Проверить транзакции"
        case .importSelected: return "Импортировать выбранные"
        case .selectAll: return "Выбрать все"
        case .deselectAll: return "Снять выделение"

        case .emailTaken: return "Аккаунт с таким email уже существует."
        case .invalidCredentials: return "Неверный email или пароль."
        case .weakPassword: return "Пароль должен содержать не менее 8 символов."
        case .emptyFields: return "Пожалуйста, заполните все поля."
        case .passwordMismatch: return "Пароли не совпадают."
        }
    }

    // MARK: - Kazakh
    var kazakh: String {
        switch self {
        case .appName: return "Fincora"
        case .cancel: return "Болдырмау"
        case .save: return "Сақтау"
        case .delete: return "Жою"
        case .done: return "Дайын"
        case .ok: return "ОК"
        case .close: return "Жабу"
        case .edit: return "Өзгерту"
        case .add: return "Қосу"
        case .seeAll: return "Барлығы"
        case .signOut: return "Шығу"
        case .loading: return "Жүктелуде..."

        case .welcomeBack: return "Қайта оралдыңыз"
        case .signInContinue: return "Жалғастыру үшін кіріңіз"
        case .email: return "Email"
        case .password: return "Құпия сөз"
        case .fullName: return "Толық аты-жөні"
        case .confirmPassword: return "Құпия сөзді растаңыз"
        case .login: return "Кіру"
        case .register: return "Тіркелу"
        case .createAccount: return "Аккаунт жасау"
        case .alreadyHaveAccount: return "Аккаунтыңыз бар ма?"
        case .dontHaveAccount: return "Аккаунтыңыз жоқ па?"
        case .forgotPassword: return "Құпия сөзді ұмыттыңыз ба?"

        case .goodMorning: return "Қайырлы таң"
        case .goodAfternoon: return "Қайырлы күн"
        case .goodEvening: return "Қайырлы кеш"
        case .totalBalance: return "Жалпы баланс"
        case .thisMonth: return "Осы ай"
        case .income: return "Кіріс"
        case .expenses: return "Шығыс"
        case .savings: return "Жинақ"
        case .cashFlow: return "Ақша ағымы"
        case .incomeVsExpenses: return "Кіріс және шығыс"
        case .spendingBreakdown: return "Шығын құрылымы"
        case .recentTransactions: return "Соңғы транзакциялар"
        case .noExpensesYet: return "Шығын жоқ — транзакция қосыңыз!"

        case .transactions: return "Транзакциялар"
        case .allTransactions: return "Барлық транзакциялар"
        case .addTransaction: return "Транзакция қосу"
        case .filterBy: return "Сүзгі"
        case .searchTransactions: return "Транзакцияларды іздеу..."
        case .noTransactions: return "Транзакция табылмады"
        case .title: return "Атауы"
        case .amount: return "Сома"
        case .category: return "Санат"
        case .date: return "Күні"
        case .type: return "Түрі"
        case .expense: return "Шығын"
        case .incomeLabel: return "Кіріс"
        case .note: return "Жазба"
        case .saveTransaction: return "Транзакцияны сақтау"
        case .deleteTransaction: return "Транзакцияны жою"
        case .editTransaction: return "Транзакцияны өзгерту"

        case .budget: return "Бюджет"
        case .stayOnTrack: return "Шығындарыңызды бақылаңыз"
        case .budgetVsActual: return "Бюджет және нақты"
        case .monthlySpending: return "Айлық шығын санаттары бойынша"
        case .budgetLabel: return "Бюджет"
        case .spent: return "Жұмсалды"
        case .left: return "Қалды"
        case .categories: return "Санаттар"
        case .setBudget: return "Бюджет белгілеу"
        case .monthlyLimit: return "Айлық шек"
        case .saveBudget: return "Бюджетті сақтау"

        case .goals: return "Мақсаттар"
        case .dreamBig: return "Үлкен арман, ақылды жинақ"
        case .progressOverview: return "Прогресс шолу"
        case .noGoals: return "Мақсат жоқ"
        case .noGoalsDesc: return "Алғашқы жинақ мақсатыңызды қойыңыз!"
        case .createGoal: return "Мақсат жасау"
        case .newGoal: return "Жаңа мақсат"
        case .pickEmoji: return "Эмодзи таңдаңыз"
        case .colorLabel: return "Түс"
        case .goalName: return "Мақсат атауы"
        case .goalNamePlaceholder: return "Мысалы, Резервтік қор"
        case .targetAmount: return "Мақсат (₸)"
        case .savedAmount: return "Жиналды (₸)"
        case .deadline: return "Мерзімі"
        case .goalAchieved: return "Мақсатқа жеттіңіз! 🎉"
        case .moreToGoal: return "мақсатқа жету үшін қалды"

        case .aiAdvisor: return "ЖИ-кеңесші"
        case .askAnything: return "Қаржыңыз туралы кез келген сұрақ қойыңыз"
        case .typeMessage: return "Хабарлама жазыңыз..."

        case .account: return "Аккаунт"
        case .profile: return "Профиль"
        case .editName: return "Атауды өзгерту"
        case .enterNewName: return "Жаңа атауыңызды енгізіңіз."
        case .notifications: return "Хабарландырулар"
        case .budgetAlerts: return "Бюджет ескертулері"
        case .weeklyReport: return "Апталық есеп"
        case .appearance: return "Сыртқы түр"
        case .language: return "Тіл"
        case .darkMode: return "Қараңғы режим"
        case .memberSince: return "Мүше болған күн"
        case .totalTransactions: return "Барлық транзакция"
        case .signOutConfirm: return "Шығасыз ба?"
        case .signOutMessage: return "Деректерге кіру үшін қайта кіруіңіз керек болады."
        case .deleteAccount: return "Аккаунтты жою"
        case .deleteAccountConfirm: return "Аккаунтты жою?"
        case .deleteAccountMessage: return "Бұл аккаунтыңыз бен барлық деректерді мәңгілікке жояды. Бұл әрекетті кері қайтаруға болмайды."
        case .dangerZone: return "Қауіпті аймақ"
        case .preferences: return "Баптаулар"
        case .support: return "Қолдау"
        case .helpCenter: return "Анықтама орталығы"
        case .privacyPolicy: return "Құпиялылық саясаты"
        case .version: return "Нұсқа"

        case .uploadStatement: return "Үзінді жүктеу"
        case .importTransactions: return "Транзакцияларды импорттау"
        case .selectFile: return "PDF файлын таңдаңыз"
        case .parsing: return "Үзінді өңделуде..."
        case .reviewTransactions: return "Транзакцияларды тексеру"
        case .importSelected: return "Таңдалғандарды импорттау"
        case .selectAll: return "Барлығын таңдау"
        case .deselectAll: return "Таңдауды алу"

        case .emailTaken: return "Бұл email-мен аккаунт бұрыннан бар."
        case .invalidCredentials: return "Қате email немесе құпия сөз."
        case .weakPassword: return "Құпия сөз кем дегенде 8 таңбадан тұруы керек."
        case .emptyFields: return "Барлық өрістерді толтырыңыз."
        case .passwordMismatch: return "Құпия сөздер сәйкес келмейді."
        }
    }
}

// MARK: - Convenience View Extension
extension View {
    func loc(_ key: LocalizedKey) -> String {
        // This is a helper for non-view contexts; use @EnvironmentObject in views
        return key.english
    }
}
