interface SelectOption {
  value: string;
  label: string;
}

interface SingleSelectProps {
  label: string;
  options: SelectOption[];
  value: string;
  onChange: (value: string) => void;
  multiple?: false;
  placeholder?: string;
  error?: string;
  optional?: boolean;
  className?: string;
}

interface MultiSelectProps {
  label: string;
  options: SelectOption[];
  value: string[];
  onChange: (value: string[]) => void;
  multiple: true;
  placeholder?: string;
  error?: string;
  optional?: boolean;
  className?: string;
}

type SelectProps = SingleSelectProps | MultiSelectProps;

export default function Select(props: SelectProps) {
  const { label, options, error, optional, className = "" } = props;

  return (
    <div className={className}>
      <label className="block text-sm font-medium text-secondary mb-2">
        {label}
        {optional && (
          <span className="ml-1 text-slate-400 dark:text-slate-500 font-normal">
            (optional)
          </span>
        )}
      </label>

      {props.multiple ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {options.map((opt) => {
            const selected = props.value.includes(opt.value);
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  if (selected) {
                    props.onChange(
                      props.value.filter((v) => v !== opt.value),
                    );
                  } else {
                    props.onChange([...props.value, opt.value]);
                  }
                }}
                className={`px-3 py-2 rounded-lg text-sm font-medium border transition-colors duration-200 cursor-pointer text-center ${
                  selected
                    ? "bg-primary text-white border-primary"
                    : "bg-card text-secondary border-border hover:border-primary dark:hover:border-primary"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      ) : (
        <select
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          className={`block w-full rounded-md border bg-card px-3 py-2 text-sm text-secondary transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary cursor-pointer ${
            error ? "border-danger" : "border-border"
          }`}
        >
          {props.placeholder && (
            <option value="" disabled>
              {props.placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      )}

      {error && <p className="mt-1.5 text-xs text-danger">{error}</p>}
    </div>
  );
}
