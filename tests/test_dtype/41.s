
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n9>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001608: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001614: 84888102
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001618: F4040000 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001624: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001628: 82070907
	s_lshl_b64 s[2:3], s[2:3], 3                               // 00000000162C: 84828302
	global_load_u16 v1, v0, s[6:7]                             // 000000001630: DC4A0000 01060000
	s_add_u32 s0, s0, s2                                       // 000000001638: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000163C: 82010301
	s_load_b64 s[0:1], s[0:1], null                            // 000000001640: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001648: BF89FC07
	s_xor_b32 s2, s0, s1                                       // 00000000164C: 8D020100
	s_cls_i32 s3, s1                                           // 000000001650: BE830C01
	s_ashr_i32 s2, s2, 31                                      // 000000001654: 86029F02
	s_add_i32 s3, s3, -1                                       // 000000001658: 8103C103
	s_add_i32 s2, s2, 32                                       // 00000000165C: 8102A002
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001660: BF870499
	s_min_u32 s2, s3, s2                                       // 000000001664: 89820203
	s_lshl_b64 s[0:1], s[0:1], s2                              // 000000001668: 84800200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000166C: BF870499
	s_min_u32 s0, s0, 1                                        // 000000001670: 89808100
	s_or_b32 s0, s1, s0                                        // 000000001674: 8C000001
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001678: BF870009
	v_cvt_f32_i32_e32 v2, s0                                   // 00000000167C: 7E040A00
	s_sub_i32 s0, 32, s2                                       // 000000001680: 818002A0
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000001684: BF870481
	v_ldexp_f32 v2, v2, s0                                     // 000000001688: D71C0002 00000102
	s_add_u32 s0, s4, s8                                       // 000000001690: 80000804
	s_addc_u32 s1, s5, s9                                      // 000000001694: 82010905
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001698: BF8700A1
	v_cvt_f16_f32_e32 v2, v2                                   // 00000000169C: 7E041502
	s_waitcnt vmcnt(0)                                         // 0000000016A0: BF8903F7
	v_mul_f16_e32 v1, v1, v2                                   // 0000000016A4: 6A020501
	global_store_b16 v0, v1, s[0:1]                            // 0000000016A8: DC660000 00000100
	s_nop 0                                                    // 0000000016B0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016B4: BFB60003
	s_endpgm                                                   // 0000000016B8: BFB00000
