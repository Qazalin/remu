
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n68>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001610: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001618: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s6, s6, s15                                      // 000000001620: 80060F06
	s_addc_u32 s7, s7, s3                                      // 000000001624: 82070307
	global_load_i8 v1, v0, s[6:7]                              // 000000001628: DC460000 01060000
	s_lshl_b64 s[6:7], s[2:3], 3                               // 000000001630: 84868302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001634: BF8704D9
	s_add_u32 s0, s0, s6                                       // 000000001638: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000163C: 82010701
	s_load_b64 s[0:1], s[0:1], null                            // 000000001640: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001648: BF89FC07
	s_clz_i32_u32 s6, s1                                       // 00000000164C: BE860A01
	s_min_u32 s6, s6, 32                                       // 000000001650: 8986A006
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001654: BF870499
	s_lshl_b64 s[0:1], s[0:1], s6                              // 000000001658: 84800600
	s_min_u32 s0, s0, 1                                        // 00000000165C: 89808100
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001660: BF870499
	s_or_b32 s0, s1, s0                                        // 000000001664: 8C000001
	v_cvt_f32_u32_e32 v2, s0                                   // 000000001668: 7E040C00
	s_sub_i32 s0, 32, s6                                       // 00000000166C: 818006A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001670: BF870481
	v_ldexp_f32 v2, v2, s0                                     // 000000001674: D71C0002 00000102
	s_lshl_b64 s[0:1], s[2:3], 1                               // 00000000167C: 84808102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001680: BF8700A9
	s_add_u32 s0, s4, s0                                       // 000000001684: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001688: 82010105
	v_cvt_f16_f32_e32 v2, v2                                   // 00000000168C: 7E041502
	s_waitcnt vmcnt(0)                                         // 000000001690: BF8903F7
	v_cvt_f16_i16_e32 v1, v1                                   // 000000001694: 7E02A301
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001698: BF870001
	v_mul_f16_e32 v1, v1, v2                                   // 00000000169C: 6A020501
	global_store_b16 v0, v1, s[0:1]                            // 0000000016A0: DC660000 00000100
	s_nop 0                                                    // 0000000016A8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016AC: BFB60003
	s_endpgm                                                   // 0000000016B0: BFB00000
