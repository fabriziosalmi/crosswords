# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depend on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### For Critical Vulnerabilities

If you discover a critical security vulnerability, please **DO NOT** open a public issue. Instead:

1. **Email us directly**: Send details to fabrizio.salmi@gmail.com
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (if you have them)

We will respond within 48 hours and work with you to understand and address the issue.

### For Non-Critical Issues

For non-critical security issues or general security concerns:

1. Open an issue on GitHub with the label `security`
2. Provide detailed information about the concern
3. We will review and respond within 5 business days

## Security Best Practices

When using this application:

### Production Deployment
- Use HTTPS/TLS for all connections
- Keep dependencies up to date (`npm audit` and `npm update`)
- Use environment variables for sensitive configuration
- Enable rate limiting in nginx
- Follow the security headers configuration in nginx.conf
- Run containers with least privilege
- Regularly update Docker images

### Development
- Never commit sensitive data (API keys, passwords, etc.)
- Use `.env` files for local configuration (not committed to git)
- Review code changes for security implications
- Test security headers and CORS policies

## Security Updates

- Security updates will be released as soon as possible
- Critical vulnerabilities will be patched within 24-48 hours
- Users will be notified via GitHub releases and security advisories

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged (with permission) in our security advisories.

Thank you for helping keep Crossword Generator and its users safe!
