
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n21>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_u8 v1, v0, s[2:3]                              // 000000001620: DC420000 01020000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001628: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000162C: BF8700C9
	s_add_u32 s0, s0, s2                                       // 000000001630: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001634: 82010301
	s_waitcnt vmcnt(0)                                         // 000000001638: BF8903F7
	v_cvt_f32_ubyte0_e32 v1, v1                                // 00000000163C: 7E022301
	v_mul_f32_e32 v1, 0x3f317218, v1                           // 000000001640: 100202FF 3F317218
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001648: BF870091
	v_mul_f32_e32 v1, 0x3fb8aa3b, v1                           // 00000000164C: 100202FF 3FB8AA3B
	v_add_f32_e32 v1, 0, v1                                    // 000000001654: 06020280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001658: BF870001
	v_exp_f32_e32 v1, v1                                       // 00000000165C: 7E024B01
	global_store_b32 v0, v1, s[0:1]                            // 000000001660: DC6A0000 00000100
	s_nop 0                                                    // 000000001668: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000166C: BFB60003
	s_endpgm                                                   // 000000001670: BFB00000
