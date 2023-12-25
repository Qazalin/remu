
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10_10>:
	s_mul_i32 s2, s15, 10                                      // 000000001600: 96028A0F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_add_i32 s2, s2, s14                                      // 00000000160C: 81020E02
	v_mov_b32_e32 v1, 0                                        // 000000001610: 7E020280
	s_mul_hi_i32 s3, s2, 0x2e8ba2e9                            // 000000001614: 9703FF02 2E8BA2E9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000161C: BF8704A9
	s_lshr_b32 s4, s3, 31                                      // 000000001620: 85049F03
	s_ashr_i32 s3, s3, 1                                       // 000000001624: 86038103
	s_add_i32 s3, s3, s4                                       // 000000001628: 81030403
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000162C: BF870499
	s_mul_i32 s3, s3, 11                                       // 000000001630: 96038B03
	s_sub_i32 s3, s2, s3                                       // 000000001634: 81830302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001638: BF870009
	s_cmp_lt_i32 s3, 1                                         // 00000000163C: BF048103
	s_cselect_b32 s4, -1, 0                                    // 000000001640: 980480C1
	s_ashr_i32 s3, s2, 31                                      // 000000001644: 86039F02
	v_cndmask_b32_e64 v0, 0, 1.0, s4                           // 000000001648: D5010000 0011E480
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001650: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001654: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001658: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000165C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001660: DC6A0000 00000001
	s_nop 0                                                    // 000000001668: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000166C: BFB60003
	s_endpgm                                                   // 000000001670: BFB00000
